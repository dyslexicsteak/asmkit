//! Windows x64 unwind info, which lets Windows walk the stack through
//! generated functions.
//!
//! Windows walks the stack whenever an exception or a Rust panic unwinds, and
//! whenever a debugger, a profiler, or a minidump reads it. It can only step
//! past a frame whose function it finds in a function table, and it finds the
//! process's compiled code in the tables of its executables. Generated code
//! isn't in any of them, so without unwind info a panic can't pass through a
//! generated frame, and the process aborts instead. Debuggers and profilers
//! lose the stack there too.
//!
//! A function's unwind info describes its prologue, which is what Windows
//! undoes to step out of the function. An [`UnwindInfo`] collects the
//! prologue's operations while the assembler emits them, and
//! [`UnwindInfo::encode`] turns them into the `UNWIND_INFO` that Windows reads.
//! Place that 4-aligned after the code, then a [`RuntimeFunction`] that
//! locates both, also 4-aligned, and register it with [`add_function_table`].
//!
//! ```
//! use asmkit::x86::unwind::{UnwindInfo, UnwindOp};
//! use asmkit::x86::*;
//! use asmkit::{Arch, CodeBuffer, Environment};
//!
//! let mut buffer = CodeBuffer::new(Environment::new(Arch::X64));
//! let mut unwind = UnwindInfo::new();
//! {
//!     let mut asm = Assembler::new(&mut buffer);
//!     asm.push(RBX);
//!     unwind.record(asm.data().len(), UnwindOp::Push(RBX))?;
//!     asm.sub(RSP, 40);
//!     unwind.record(asm.data().len(), UnwindOp::Alloc(40))?;
//!     // ... the body of the function ...
//!     asm.add(RSP, 40);
//!     asm.pop(RBX);
//!     asm.ret();
//! }
//! let code = buffer.finish().expect("the function should assemble");
//! let info = unwind.encode();
//! # let _ = (code, info);
//! # Ok::<(), asmkit::x86::unwind::UnwindError>(())
//! ```
//!
//! The format is Microsoft's, in [x64 exception handling].
//!
//! [x64 exception handling]: https://learn.microsoft.com/en-us/cpp/build/exception-handling-x64

use alloc::vec::Vec;
use core::fmt;

use super::operands::{Gpq, Xmm};

#[cfg(test)]
mod tests;

/// The unwind code of a push of a general-purpose register.
const UWOP_PUSH_NONVOL: u8 = 0;
/// The unwind code of a stack allocation whose size, in 8s, takes the next
/// slot.
const UWOP_ALLOC_LARGE: u8 = 1;
/// The unwind code of a stack allocation of 8 to 128 bytes, whose size is in
/// its operand.
const UWOP_ALLOC_SMALL: u8 = 2;
/// The unwind code of a save of an `xmm` register, whose offset, in 16s,
/// takes the next slot.
const UWOP_SAVE_XMM128: u8 = 8;

/// One operation of a function's prologue, as `UNWIND_INFO` records it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnwindOp {
    /// A `push` of a general-purpose register. Its encoding number is its id,
    /// so `RBX` is 3.
    Push(Gpq),
    /// A `sub rsp` of this many bytes, which must be a multiple of 8 below
    /// 512 KiB.
    Alloc(u32),
    /// A save of `xmm` register `register` to `[rsp + offset]`, where `rsp` is
    /// its value after the prologue's allocation. The register must be `xmm6`
    /// to `xmm15`, the nonvolatile ones, and the offset a multiple of 16 below
    /// 1 MiB.
    SaveXmm {
        /// The register saved, from `xmm6` to `xmm15`.
        register: Xmm,
        /// The save's byte offset from `rsp`.
        offset: u32,
    },
}

/// Why [`UnwindInfo::record`] refused an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnwindError {
    /// The operation ends past byte 255, which `UNWIND_INFO` can't record.
    EndOutOfRange {
        /// The end offset given.
        end: usize,
    },
    /// The operation doesn't end after the one recorded before it.
    EndNotAfterPrevious {
        /// The end offset given.
        end: usize,
        /// The end offset of the operation recorded before.
        previous: u8,
    },
    /// The prologue would need more unwind slots than `CountOfCodes` holds.
    TooManyCodes,
    /// A push of a register that isn't a general-purpose one, as its id
    /// isn't below 16.
    InvalidRegister {
        /// The register's id.
        id: u32,
    },
    /// An allocation that isn't a multiple of 8, is zero, or is 512 KiB or
    /// more.
    InvalidAllocation {
        /// The size given, in bytes.
        size: u32,
    },
    /// A save of an `xmm` register that isn't one of `xmm6` to `xmm15`.
    InvalidXmmRegister {
        /// The register's id.
        id: u32,
    },
    /// A save at an offset that isn't a multiple of 16, or is 1 MiB or more.
    InvalidSaveOffset {
        /// The offset given, in bytes.
        offset: u32,
    },
}

impl fmt::Display for UnwindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            UnwindError::EndOutOfRange { end } => {
                write!(f, "prologue operation ends at byte {end}, past byte 255")
            }
            UnwindError::EndNotAfterPrevious { end, previous } => write!(
                f,
                "prologue operation ends at byte {end}, not after byte {previous}"
            ),
            UnwindError::TooManyCodes => {
                write!(f, "prologue needs more than 255 unwind codes")
            }
            UnwindError::InvalidRegister { id } => {
                write!(f, "register id {id} is not a general-purpose register")
            }
            UnwindError::InvalidAllocation { size } => write!(
                f,
                "allocation of {size} bytes is not a multiple of 8 below 512 KiB"
            ),
            UnwindError::InvalidXmmRegister { id } => {
                write!(f, "xmm{id} is not a nonvolatile register, xmm6 to xmm15")
            }
            UnwindError::InvalidSaveOffset { offset } => write!(
                f,
                "save at offset {offset} is not a multiple of 16 below 1 MiB"
            ),
        }
    }
}

impl core::error::Error for UnwindError {}

/// The prologue of a generated function, which [`encode`] turns into the
/// `UNWIND_INFO` that Windows reads.
///
/// [`encode`]: UnwindInfo::encode
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct UnwindInfo {
    /// Each operation, with the offset of the byte after its instruction, in
    /// the order the prologue runs them.
    ops: Vec<(u8, UnwindOp)>,
    /// The unwind slots the operations take, which `CountOfCodes` counts.
    codes: u8,
}

impl UnwindInfo {
    /// Returns a prologue with no operations, which is a leaf function's.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ops: Vec::new(),
            codes: 0,
        }
    }

    /// Records `op`, whose instruction ends just before byte `end_offset` of
    /// the function. Call it right after emitting the instruction, with the
    /// assembler's length as `end_offset`.
    ///
    /// # Errors
    ///
    /// Returns [`UnwindError::EndOutOfRange`] if `end_offset` is past byte
    /// 255, [`UnwindError::EndNotAfterPrevious`] if it isn't after the
    /// previous operation's, [`UnwindError::InvalidRegister`] if `op` pushes a
    /// register that isn't general-purpose, [`UnwindError::InvalidAllocation`]
    /// if it allocates 0 bytes, 512 KiB or more, or a size that isn't a
    /// multiple of 8, [`UnwindError::InvalidXmmRegister`] if it saves a
    /// register other than `xmm6` to `xmm15`, [`UnwindError::InvalidSaveOffset`]
    /// if it saves at an offset of 1 MiB or more or one that isn't a multiple
    /// of 16, or [`UnwindError::TooManyCodes`] if the prologue would need more
    /// than 255 unwind slots. The info is unchanged after an error.
    pub fn record(&mut self, end_offset: usize, op: UnwindOp) -> Result<(), UnwindError> {
        let end =
            u8::try_from(end_offset).map_err(|_| UnwindError::EndOutOfRange { end: end_offset })?;
        if let Some(&(previous, _)) = self.ops.last().filter(|&&(previous, _)| end <= previous) {
            return Err(UnwindError::EndNotAfterPrevious {
                end: end_offset,
                previous,
            });
        }
        // The slots the operation takes, which `CountOfCodes` counts.
        let slots = match op {
            UnwindOp::Push(register) => {
                if register.id() >= 16 {
                    return Err(UnwindError::InvalidRegister { id: register.id() });
                }
                1
            }
            UnwindOp::Alloc(size) => {
                if size % 8 != 0 || size == 0 || size >= 512 * 1024 {
                    return Err(UnwindError::InvalidAllocation { size });
                }
                if size <= 128 { 1 } else { 2 }
            }
            UnwindOp::SaveXmm { register, offset } => {
                if !(6..16).contains(&register.id()) {
                    return Err(UnwindError::InvalidXmmRegister { id: register.id() });
                }
                if offset % 16 != 0 || offset >= 16 * 0x1_0000 {
                    return Err(UnwindError::InvalidSaveOffset { offset });
                }
                2
            }
        };
        // `CountOfCodes` is a byte, so the slots can't pass 255.
        self.codes = self
            .codes
            .checked_add(slots)
            .ok_or(UnwindError::TooManyCodes)?;
        self.ops.push((end, op));
        Ok(())
    }

    /// Returns the `UNWIND_INFO` of the prologue: version 1, the prologue's
    /// size, its unwind codes in reverse order, and padding.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        // Each code is a byte offset in the low byte, then its operation in
        // the next four bits and its operand in the top four.
        let slot =
            |end: u8, op: u8, info: u8| u16::from(end) | u16::from(op) << 8 | u16::from(info) << 12;
        let mut slots = Vec::with_capacity(usize::from(self.codes) + 1);
        for &(end, op) in self.ops.iter().rev() {
            match op {
                UnwindOp::Push(register) => {
                    // `record` keeps the id below 16, so it fits the operand.
                    slots.push(slot(end, UWOP_PUSH_NONVOL, register.id() as u8));
                }
                UnwindOp::Alloc(size) if size <= 128 => {
                    // `record` keeps the size in 8..=128, so the operand is 0..=15.
                    let info = ((size - 8) / 8) as u8;
                    slots.push(slot(end, UWOP_ALLOC_SMALL, info));
                }
                UnwindOp::Alloc(size) => {
                    slots.push(slot(end, UWOP_ALLOC_LARGE, 0));
                    // `record` keeps the size below 512 KiB, so it fits 16 bits.
                    slots.push((size / 8) as u16);
                }
                UnwindOp::SaveXmm { register, offset } => {
                    slots.push(slot(end, UWOP_SAVE_XMM128, register.id() as u8));
                    // `record` keeps the offset below 1 MiB, so it fits 16 bits.
                    slots.push((offset / 16) as u16);
                }
            }
        }
        debug_assert_eq!(slots.len(), usize::from(self.codes));

        let prologue = self.ops.last().map_or(0, |&(end, _)| end);
        let mut info = Vec::with_capacity(4 + 2 * slots.len() + 2);
        info.extend([1, prologue, self.codes, 0]);
        // The slot array has an even length so that the info's size stays a
        // multiple of 4, as Windows expects. An odd count gets an all-zero
        // slot as padding, and `CountOfCodes`, which is `self.codes`, leaves
        // that slot out. If `CountOfCodes` counted the padding, Windows would
        // decode the zero slot as `UWOP_PUSH_NONVOL` of register 0, which is
        // `rax`. Windows would then undo a push of `rax` that never happened.
        // It would load `rax` from the stack and leave the stack pointer 8
        // bytes off.
        if slots.len() % 2 == 1 {
            slots.push(0);
        }
        info.extend(slots.iter().flat_map(|slot| slot.to_le_bytes()));
        info
    }
}

/// A function's entry in a function table, `RUNTIME_FUNCTION` in Windows and
/// `IMAGE_RUNTIME_FUNCTION_ENTRY` in its headers.
///
/// Each field is a relative virtual address, an offset from the base address
/// the table is registered with. `begin` and `end` bound the function's code,
/// and `unwind_data` locates its `UNWIND_INFO`.
///
/// Entries compare by `begin` first, so sorting a table orders it by `begin`,
/// as [`add_function_table`] asks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(C)]
pub struct RuntimeFunction {
    /// The offset of the function's first byte.
    pub begin: u32,
    /// The offset just after the function's last byte.
    pub end: u32,
    /// The offset of the function's [`UnwindInfo::encode`] output.
    pub unwind_data: u32,
}

// Windows reads this layout, so it must never change.
const _: () = assert!(core::mem::size_of::<RuntimeFunction>() == 12);

#[cfg(all(windows, target_arch = "x86_64", feature = "jit"))]
mod native {
    use super::RuntimeFunction;

    unsafe extern "system" {
        pub(super) fn RtlAddFunctionTable(
            function_table: *const RuntimeFunction,
            entry_count: u32,
            base_address: u64,
        ) -> u8;
        pub(super) fn RtlDeleteFunctionTable(function_table: *const RuntimeFunction) -> u8;
    }
}

/// Registers a function table with Windows, so Windows can walk the stack
/// through the functions in it. Returns `false` if Windows refuses the table,
/// which happens when it has no memory to record it.
///
/// Keep the table's entries sorted by `begin`. Microsoft documents that only
/// for growable tables, and a table of one entry is always sorted.
///
/// # Safety
///
/// - `base` must be the address that the table's offsets count from, so every
///   function and its unwind info must lie less than 4 GiB above it.
/// - The table, its functions' code, and their unwind info must stay valid
///   for reads and unmoved until [`delete_function_table`] removes the table.
#[cfg(all(windows, target_arch = "x86_64", feature = "jit"))]
pub unsafe fn add_function_table(table: &[RuntimeFunction], base: *const u8) -> bool {
    let Ok(count) = u32::try_from(table.len()) else {
        return false;
    };
    // SAFETY: The caller guarantees the table and its base, as this function
    // documents, and `table` holds `count` entries.
    unsafe { native::RtlAddFunctionTable(table.as_ptr(), count, base.addr() as u64) != 0 }
}

/// Removes a function table that [`add_function_table`] registered, after
/// which Windows no longer reads it. Returns `false` if Windows doesn't know
/// the table.
///
/// # Safety
///
/// - `table` must be the pointer that [`add_function_table`] was given, and
///   the table must not have been removed already.
/// - No thread may be running a function in the table, or have one on its
///   stack, because Windows can no longer unwind through it.
#[cfg(all(windows, target_arch = "x86_64", feature = "jit"))]
pub unsafe fn delete_function_table(table: *const RuntimeFunction) -> bool {
    // SAFETY: The caller guarantees `table` is registered and not yet removed.
    unsafe { native::RtlDeleteFunctionTable(table) != 0 }
}

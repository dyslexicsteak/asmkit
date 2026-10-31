//! Tests of Windows x64 unwind info. The encoding is checked byte for byte,
//! against the format in Microsoft's x64 exception handling docs. The
//! Windows-only tests run generated functions, and check that Windows unwinds
//! through them, as an exception or a panic does.

use core::mem;

use crate::x86::regs::{RBX, RSI, XMM6, gpq, xmm};

use super::{UnwindError, UnwindInfo, UnwindOp};

/// Checks the bytes of an `UNWIND_INFO` with every kind of unwind code, one
/// per operation.
#[test]
fn encodes_unwind_info() {
    let mut unwind = UnwindInfo::new();
    unwind
        .record(1, UnwindOp::Push(RBX))
        .expect("a push of a general-purpose register should be recorded");
    unwind
        .record(2, UnwindOp::Push(RSI))
        .expect("a push of a general-purpose register should be recorded");
    unwind
        .record(6, UnwindOp::Alloc(40))
        .expect("an allocation of a multiple of 8 should be recorded");
    unwind
        .record(
            11,
            UnwindOp::SaveXmm {
                register: XMM6,
                offset: 32,
            },
        )
        .expect("a save of xmm6 at a multiple of 16 should be recorded");

    // Version 1, a prologue of 11 bytes, and a count of 5 slots. Then come the
    // codes, last operation first, so the save and its offset in 16s, the
    // small allocation of (40 - 8) / 8, and the pushes of rsi and rbx.
    let expected = [
        1, 11, 5, 0, 0x0B, 0x68, 2, 0, 0x06, 0x42, 0x02, 0x60, 0x01, 0x30, 0, 0,
    ];
    assert_eq!(unwind.encode(), expected);
}

/// Checks that an allocation above 128 bytes takes a second slot of its size
/// in 8s.
#[test]
fn encodes_a_large_allocation() {
    let mut unwind = UnwindInfo::new();
    unwind
        .record(7, UnwindOp::Alloc(256))
        .expect("an allocation of a multiple of 8 should be recorded");
    assert_eq!(unwind.encode(), [1, 7, 2, 0, 0x07, 0x01, 32, 0]);
}

/// Checks that a leaf function, with no prologue, has no codes at all.
#[test]
fn encodes_a_leaf() {
    assert_eq!(UnwindInfo::new().encode(), [1, 0, 0, 0]);
}

/// Checks that an odd number of slots is padded to an even one, and that the
/// count leaves the padding out. A count that included it would make Windows
/// read the zero slot as a `push` and unwind through the wrong frame.
#[test]
fn count_leaves_out_the_padding_slot() {
    let mut unwind = UnwindInfo::new();
    unwind
        .record(1, UnwindOp::Push(RBX))
        .expect("a push of a general-purpose register should be recorded");

    // One real slot, `push rbx` at byte 1, then one padding slot of zeros.
    let info = unwind.encode();
    assert_eq!(info, [1, 1, 1, 0, 0x01, 0x30, 0, 0]);
    assert_eq!(info[2], 1, "the count should leave out the padding");

    // Three real slots are padded to four, and the count is still three.
    let mut three = UnwindInfo::new();
    three
        .record(1, UnwindOp::Push(RBX))
        .expect("a push of a general-purpose register should be recorded");
    three
        .record(2, UnwindOp::Push(RSI))
        .expect("a push of a general-purpose register should be recorded");
    three
        .record(6, UnwindOp::Alloc(40))
        .expect("an allocation of a multiple of 8 should be recorded");
    let info = three.encode();
    assert_eq!(info.len(), 4 + 4 * 2);
    assert_eq!(info[2], 3, "the count should leave out the padding");
    assert_eq!(
        &info[info.len() - 2..],
        [0, 0],
        "the padding should be zero"
    );
}

/// Checks that operations that end out of order are refused, and that a
/// refused one isn't recorded.
#[test]
fn refuses_codes_out_of_order() {
    let mut unwind = UnwindInfo::new();
    unwind
        .record(4, UnwindOp::Push(RBX))
        .expect("a push of a general-purpose register should be recorded");
    assert_eq!(
        unwind.record(4, UnwindOp::Push(RSI)),
        Err(UnwindError::EndNotAfterPrevious {
            end: 4,
            previous: 4
        })
    );
    assert_eq!(unwind.encode(), [1, 4, 1, 0, 0x04, 0x30, 0, 0]);
}

/// Checks that each operation outside what `UNWIND_INFO` records is refused.
#[test]
fn refuses_invalid_operations() {
    let refused = |end: usize, op: UnwindOp| UnwindInfo::new().record(end, op);

    assert_eq!(
        refused(256, UnwindOp::Push(RBX)),
        Err(UnwindError::EndOutOfRange { end: 256 })
    );
    assert_eq!(
        refused(4, UnwindOp::Push(gpq(16))),
        Err(UnwindError::InvalidRegister { id: 16 })
    );
    for size in [0, 12, 512 * 1024, 512 * 1024 + 8] {
        assert_eq!(
            refused(4, UnwindOp::Alloc(size)),
            Err(UnwindError::InvalidAllocation { size }),
            "an allocation of {size} bytes"
        );
    }
    assert_eq!(
        refused(
            4,
            UnwindOp::SaveXmm {
                register: xmm(5),
                offset: 16,
            }
        ),
        Err(UnwindError::InvalidXmmRegister { id: 5 })
    );
    for offset in [8, 16 * 0x1_0000] {
        assert_eq!(
            refused(
                4,
                UnwindOp::SaveXmm {
                    register: XMM6,
                    offset,
                }
            ),
            Err(UnwindError::InvalidSaveOffset { offset }),
            "a save at offset {offset}"
        );
    }
}

/// Checks that a prologue needing more than 255 slots is refused, which
/// `CountOfCodes` can't count. Each save takes two slots, so the 128th, which
/// makes 256, is the one refused, and the 127 before it are kept.
#[test]
fn refuses_too_many_codes() {
    let save = UnwindOp::SaveXmm {
        register: XMM6,
        offset: 16,
    };
    let mut unwind = UnwindInfo::new();
    for end in 1..=127 {
        unwind
            .record(end, save)
            .expect("127 saves should fit in 254 slots");
    }
    assert_eq!(unwind.record(128, save), Err(UnwindError::TooManyCodes));
    assert_eq!(unwind.encode()[2], 254, "the refused save shouldn't count");
}

/// Checks that a `RuntimeFunction` is the 12-byte entry that Windows reads.
#[test]
fn runtime_function_is_twelve_bytes() {
    assert_eq!(mem::size_of::<super::RuntimeFunction>(), 12);
}

#[cfg(all(windows, target_arch = "x86_64", feature = "jit"))]
mod native {
    extern crate std;

    use std::{
        cell::Cell,
        panic::{self, AssertUnwindSafe},
        ptr,
    };

    use alloc::boxed::Box;

    use crate::x86::regs::{RAX, RBX, RCX, RDI, RDX, RSI, RSP, XMM6};
    use crate::x86::{
        AddEmitter, Assembler, CallEmitter, Gpq, MovEmitter, MovqEmitter, MovupsEmitter,
        PopEmitter, PushEmitter, RetEmitter, SubEmitter, xmmword_ptr,
    };
    use crate::{Arch, CodeBuffer, Environment, JitAllocator, Span};
    use core::mem;
    use windows::Win32::System::Diagnostics::Debug::{
        CONTEXT, RtlCaptureContext, RtlLookupFunctionEntry, RtlVirtualUnwind, UNW_FLAG_NHANDLER,
    };

    use super::super::{
        RuntimeFunction, UnwindInfo, UnwindOp, add_function_table, delete_function_table,
    };

    /// Installs `code`, whose prologue `unwind` describes, in executable memory
    /// with its unwind info and a function table, and registers the table.
    ///
    /// The image is the code, then the unwind info, then the table entry, each
    /// 4-aligned. Its RVAs count from the start of the code.
    fn install(code: &[u8], unwind: &UnwindInfo) -> Installed {
        let info = unwind.encode();
        let at_info = code.len().next_multiple_of(4);
        let at_table = (at_info + info.len()).next_multiple_of(4);
        let len = at_table + mem::size_of::<RuntimeFunction>();

        let mut image = alloc::vec![0u8; len];
        image[..code.len()].copy_from_slice(code);
        image[at_info..at_info + info.len()].copy_from_slice(&info);
        let entry = RuntimeFunction {
            begin: 0,
            end: u32::try_from(code.len()).expect("a test function should be far below 4 GiB"),
            unwind_data: u32::try_from(at_info).expect("a test function should be far below 4 GiB"),
        };
        image[at_table..at_table + 4].copy_from_slice(&entry.begin.to_le_bytes());
        image[at_table + 4..at_table + 8].copy_from_slice(&entry.end.to_le_bytes());
        image[at_table + 8..at_table + 12].copy_from_slice(&entry.unwind_data.to_le_bytes());

        let mut jit = JitAllocator::new(Default::default());
        let mut span = jit.alloc(len).expect("the function should allocate");
        // SAFETY: `span` came from `jit` and is `len` bytes, and nothing runs it
        // yet.
        unsafe { jit.copy_from_slice(&mut span, 0, &image) }.expect("the image should copy");

        // SAFETY: `at_table` is inside the span, as the image is `len` bytes, and
        // holds one entry, so the slice covers it.
        let table = unsafe { span.rx().add(at_table) }.cast::<RuntimeFunction>();
        // SAFETY: The one-entry table is in the span, and the span's base is the
        // address its RVAs count from. The span outlives the registration,
        // because `Installed` removes it before it releases the span.
        let registered =
            unsafe { add_function_table(core::slice::from_raw_parts(table, 1), span.rx()) };
        assert!(registered, "Windows should register the function table");

        Installed { jit, span, table }
    }

    /// A function installed by [`install`], whose table is registered until it
    /// drops.
    struct Installed {
        jit: Box<JitAllocator>,
        span: Span,
        table: *const RuntimeFunction,
    }

    impl Installed {
        /// Returns the function's entry, the first byte of its code.
        fn entry(&self) -> *const u8 {
            self.span.rx()
        }
    }

    impl Drop for Installed {
        fn drop(&mut self) {
            // SAFETY: `table` is the pointer that `install` registered, and it's
            // removed only here.
            let deleted = unsafe { delete_function_table(self.table) };
            if deleted {
                // SAFETY: Windows no longer reads the table, and nothing runs
                // the function, since every test drops its `Installed` after its
                // calls.
                let _ = unsafe { self.jit.release(self.span.rx()) };
            }
        }
    }

    /// A callback that a generated function calls with `value`.
    type Callback = extern "C-unwind" fn(u64) -> u64;

    /// A generated function that calls `callback` with `value`, and returns its
    /// result plus `value`.
    type Caller = unsafe extern "C-unwind" fn(callback: Callback, value: u64) -> u64;

    /// Returns twice `value`.
    extern "C-unwind" fn double(value: u64) -> u64 {
        value * 2
    }

    /// Panics.
    extern "C-unwind" fn fail(_: u64) -> u64 {
        panic!("the callback panics, as the test means it to");
    }

    std::thread_local! {
        /// The return address and `rsp` of each frame that `walk` unwinds to,
        /// from its caller outwards, or zeros past the last.
        static WALKED: Cell<[(u64, u64); 4]> = const { Cell::new([(0, 0); 4]) };
    }

    /// Unwinds up to four frames out from where it captures its context, with
    /// `RtlVirtualUnwind`, the way an exception does, and keeps where each lands
    /// in `WALKED`. The first frame it unwinds is its own, so the one that
    /// called the generated code is the second.
    extern "C-unwind" fn walk(value: u64) -> u64 {
        /// A context record 16-aligned, as Windows writes it, which the
        /// `windows` crate's type isn't.
        #[repr(C, align(16))]
        struct Aligned(CONTEXT);

        let mut aligned = Aligned(CONTEXT::default());
        let context = &mut aligned.0;
        // SAFETY: `context` is a place for a context record, and 16-aligned.
        unsafe { RtlCaptureContext(&raw mut *context) };
        let mut walked = [(0, 0); 4];
        for landed in &mut walked {
            let mut base = 0;
            // SAFETY: `RtlLookupFunctionEntry` only reads the registered tables.
            let entry = unsafe { RtlLookupFunctionEntry(context.Rip, &raw mut base, None) };
            if entry.is_null() {
                break;
            }
            let mut data = ptr::null_mut();
            let mut frame = 0;
            // SAFETY: `entry` is the function entry of `context.Rip`, and
            // `context` holds a frame that's live below this one.
            unsafe {
                RtlVirtualUnwind(
                    UNW_FLAG_NHANDLER,
                    base,
                    context.Rip,
                    entry,
                    &raw mut *context,
                    &raw mut data,
                    &raw mut frame,
                    None,
                )
            };
            *landed = (context.Rip, context.Rsp);
        }
        WALKED.set(walked);
        value
    }

    /// A frame that a generated function sets up: its `pushes` in order, then
    /// an allocation of `alloc` bytes, with `xmm6` saved at `rsp + 16`.
    struct Shape {
        pushes: &'static [Gpq],
        alloc: u32,
    }

    /// Frames of every shape that the tests unwind through. The slot counts are
    /// 4, 5, 6 and 5, so both an odd count, which needs padding, and an even one
    /// are covered, and so is a large allocation. Each keeps the stack 16-aligned
    /// at its calls.
    const SHAPES: [Shape; 4] = [
        Shape {
            pushes: &[RBX],
            alloc: 32,
        },
        Shape {
            pushes: &[RBX, RSI],
            alloc: 40,
        },
        Shape {
            pushes: &[RBX, RSI, RDI],
            alloc: 32,
        },
        Shape {
            pushes: &[RBX],
            alloc: 256,
        },
    ];

    /// Emits the prologue of `shape` and records each instruction in `unwind`.
    fn emit_prologue(asm: &mut Assembler<'_>, unwind: &mut UnwindInfo, shape: &Shape) {
        for &register in shape.pushes {
            asm.push(register);
            unwind
                .record(asm.data().len(), UnwindOp::Push(register))
                .expect("the push should record");
        }
        asm.sub(
            RSP,
            i32::try_from(shape.alloc).expect("a test frame's allocation should fit an `i32`"),
        );
        unwind
            .record(asm.data().len(), UnwindOp::Alloc(shape.alloc))
            .expect("the allocation should record");
        asm.movups(xmmword_ptr(RSP, 16), XMM6.0);
        unwind
            .record(
                asm.data().len(),
                UnwindOp::SaveXmm {
                    register: XMM6,
                    offset: 16,
                },
            )
            .expect("the save should record");
    }

    /// Emits the epilogue of `shape`, which restores what its prologue saved.
    fn emit_epilogue(asm: &mut Assembler<'_>, shape: &Shape) {
        asm.movups(XMM6, xmmword_ptr(RSP, 16));
        asm.add(
            RSP,
            i32::try_from(shape.alloc).expect("a test frame's allocation should fit an `i32`"),
        );
        for &register in shape.pushes.iter().rev() {
            asm.pop(register);
        }
        asm.ret();
    }

    /// Installs a function of `shape` that calls its callback with `value`, and
    /// returns it with the offset in its code of the return address of its
    /// `call`. The callback's result plus `value` is what it returns.
    fn caller_in(shape: &Shape) -> (Installed, u64) {
        let mut unwind = UnwindInfo::new();
        let mut buffer = CodeBuffer::new(Environment::new(Arch::X64));
        let after_call;
        {
            let mut asm = Assembler::new(&mut buffer);
            emit_prologue(&mut asm, &mut unwind, shape);
            asm.mov(RBX, RDX);
            asm.movq(XMM6, RDX);
            asm.mov(RAX, RCX);
            asm.mov(RCX, RDX);
            asm.call(RAX);
            after_call = asm.data().len() as u64;
            asm.add(RAX, RBX);
            emit_epilogue(&mut asm, shape);
        }
        let code = buffer.finish().expect("the caller should assemble");
        (install(code.data(), &unwind), after_call)
    }

    /// Returns the function of `installed` as a `Caller`.
    fn as_caller(installed: &Installed) -> Caller {
        // SAFETY: `installed` holds a function with the `Caller` signature, as
        // `caller_in` assembles it, and the pointer is the function's entry.
        unsafe { mem::transmute::<*const u8, Caller>(installed.entry()) }
    }

    /// Checks that a frame function runs from its pages and returns what it
    /// computes, so its prologue and epilogue keep `rbx` across the call.
    #[test]
    fn runs_a_frame_function() {
        let (installed, _) = caller_in(&SHAPES[1]);
        // SAFETY: `double` is a valid callback, and the function saves every
        // nonvolatile register it uses.
        let result = unsafe { as_caller(&installed)(double, 21) };
        assert_eq!(result, 63);
    }

    /// Checks that Windows finds the function's table entry from an address in
    /// its code, and nothing past it.
    #[test]
    fn registers_the_function_table() {
        let (installed, _) = caller_in(&SHAPES[1]);
        let entry = installed.entry().addr() as u64;
        let mut base = 0;

        // SAFETY: `RtlLookupFunctionEntry` only reads the registered tables.
        let found = unsafe { RtlLookupFunctionEntry(entry + 4, &raw mut base, None) };
        assert!(
            !found.is_null(),
            "the function's entry should be registered"
        );
        assert_eq!(base, entry);

        drop(installed);
        // SAFETY: As above.
        let gone = unsafe { RtlLookupFunctionEntry(entry + 4, &raw mut base, None) };
        assert_eq!(
            gone,
            ptr::null_mut(),
            "dropping the function should unregister it"
        );
    }

    /// Checks that Windows unwinds frame functions of every shape to their
    /// caller, as an exception does. For each, the walk out of the callback
    /// lands just after the generated `call`, and the walk out of the generated
    /// frame lands in Rust, at the same place for every shape, since one call
    /// site calls them all.
    #[test]
    fn unwinds_frame_functions() {
        let mut outers = alloc::vec::Vec::new();
        for (index, shape) in SHAPES.iter().enumerate() {
            let (installed, after_call) = caller_in(shape);
            // SAFETY: As in `runs_a_frame_function`.
            assert_eq!(unsafe { as_caller(&installed)(walk, 5) }, 10);

            let walked = WALKED.get();
            let after_call = installed.entry().addr() as u64 + after_call;
            let at = walked
                .iter()
                .position(|&(address, _)| address == after_call)
                .expect("the callback should return after the generated `call`");
            let (outer, _) = walked[at + 1];
            let mut base = 0;
            // SAFETY: `RtlLookupFunctionEntry` only reads the registered tables.
            let found = unsafe { RtlLookupFunctionEntry(outer, &raw mut base, None) };
            assert!(
                !found.is_null(),
                "shape {index} should unwind to a function"
            );
            assert_ne!(
                base,
                installed.entry().addr() as u64,
                "shape {index} should unwind out of the generated code"
            );
            outers.push(walked[at + 1]);
        }
        assert!(
            outers.iter().all(|&outer| outer == outers[0]),
            "every shape should unwind to the one call site: {outers:#x?}"
        );
    }

    /// Checks that a panic in a callback unwinds through the generated frame to
    /// a `catch_unwind` in Rust, which only works if Windows reads the frame's
    /// unwind info as meant, and that the function still works afterwards.
    #[test]
    fn panics_unwind_through_frames() {
        let (installed, _) = caller_in(&SHAPES[1]);
        let caller = as_caller(&installed);

        // SAFETY: As in `runs_a_frame_function`, and `fail` unwinds, which the
        // `C-unwind` ABI allows.
        let caught = panic::catch_unwind(AssertUnwindSafe(|| unsafe { caller(fail, 1) }));
        assert!(caught.is_err(), "the panic should reach `catch_unwind`");

        // SAFETY: As in `runs_a_frame_function`.
        assert_eq!(unsafe { caller(double, 2) }, 6);
    }

    /// Checks that a leaf, with no prologue, runs from its pages.
    #[test]
    fn runs_a_leaf() {
        let mut buffer = CodeBuffer::new(Environment::new(Arch::X64));
        {
            let mut asm = Assembler::new(&mut buffer);
            asm.mov(RAX, RCX);
            asm.add(RAX, RDX);
            asm.ret();
        }
        let bytes = buffer.finish().expect("the leaf should assemble");
        let installed = install(bytes.data(), &UnwindInfo::new());

        // SAFETY: The leaf adds its two arguments, touching only volatile
        // registers.
        let add = unsafe {
            mem::transmute::<*const u8, unsafe extern "C-unwind" fn(u64, u64) -> u64>(
                installed.entry(),
            )
        };
        // SAFETY: As above.
        assert_eq!(unsafe { add(40, 2) }, 42);
    }
}

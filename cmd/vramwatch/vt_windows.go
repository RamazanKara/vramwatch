//go:build windows

package main

import (
	"os"
	"syscall"
	"unsafe"
)

// enableVT turns on ENABLE_VIRTUAL_TERMINAL_PROCESSING so ANSI escape codes
// work in legacy Windows consoles. Modern Windows Terminal enables this by
// default; this is a best-effort no-op if it can't be set.
func enableVT() {
	const enableVirtualTerminalProcessing = 0x0004
	handle := syscall.Handle(os.Stdout.Fd())

	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	getConsoleMode := kernel32.NewProc("GetConsoleMode")
	setConsoleMode := kernel32.NewProc("SetConsoleMode")

	var mode uint32
	if r, _, _ := getConsoleMode.Call(uintptr(handle), uintptr(unsafe.Pointer(&mode))); r == 0 {
		return // not a console
	}
	setConsoleMode.Call(uintptr(handle), uintptr(mode|enableVirtualTerminalProcessing))
}

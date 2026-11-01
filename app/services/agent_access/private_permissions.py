"""Private filesystem permissions for local Agent Access credentials.

Windows paths receive a protected DACL for only the current process user and
SYSTEM. POSIX permissions are set by the caller and this module is a no-op.
"""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from pathlib import Path
from typing import Protocol


class PrivatePermissionError(RuntimeError):
    """A private credential path could not be protected and verified."""


class _Protector(Protocol):
    def protect(self, path: Path, *, directory: bool) -> None: ...


def protect_private_path(
    path: str | os.PathLike[str],
    *,
    directory: bool,
    windows: bool | None = None,
    security_api: _Protector | None = None,
) -> None:
    """Protect one credential path; Windows failures are always fatal.

    ``windows`` and ``security_api`` are injectable for focused tests. A
    successful Windows call means the resulting owner and protected DACL were
    read back and verified, not just that the ACL update API returned success.
    """
    if windows is None:
        windows = os.name == "nt"
    if not windows:
        return
    try:
        protector = security_api or _WindowsSecurityAPI()
        protector.protect(Path(path), directory=directory)
    except PrivatePermissionError:
        raise
    except Exception as error:
        raise PrivatePermissionError("Could not protect the local credential path") from error


class _SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]


class _TOKEN_USER(ctypes.Structure):
    _fields_ = [("User", _SID_AND_ATTRIBUTES)]


class _ACL_SIZE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("AceCount", wintypes.DWORD),
        ("AclBytesInUse", wintypes.DWORD),
        ("AclBytesFree", wintypes.DWORD),
    ]


class _ACE_HEADER(ctypes.Structure):
    _fields_ = [
        ("AceType", ctypes.c_ubyte),
        ("AceFlags", ctypes.c_ubyte),
        ("AceSize", wintypes.WORD),
    ]


class _ACCESS_ALLOWED_ACE(ctypes.Structure):
    _fields_ = [("Header", _ACE_HEADER), ("Mask", wintypes.DWORD), ("SidStart", wintypes.DWORD)]


class _WindowsSecurityAPI:
    TOKEN_QUERY = 0x0008
    TOKEN_USER_CLASS = 1
    ERROR_INSUFFICIENT_BUFFER = 122
    SE_FILE_OBJECT = 1
    OWNER_SECURITY_INFORMATION = 0x00000001
    DACL_SECURITY_INFORMATION = 0x00000004
    PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000
    SE_DACL_PROTECTED = 0x1000
    ACL_SIZE_INFORMATION_CLASS = 2
    ACCESS_ALLOWED_ACE_TYPE = 0
    FILE_ALL_ACCESS = 0x001F01FF
    OBJECT_INHERIT_ACE = 0x01
    CONTAINER_INHERIT_ACE = 0x02
    SYSTEM_SID = "S-1-5-18"

    def __init__(self) -> None:
        if os.name != "nt":
            raise PrivatePermissionError("Windows security APIs are unavailable")
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        self._declare_functions()

    def _declare_functions(self) -> None:
        k, a = self.kernel32, self.advapi32
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.CloseHandle.restype = wintypes.BOOL
        k.LocalFree.argtypes = [ctypes.c_void_p]
        k.LocalFree.restype = ctypes.c_void_p

        a.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
        a.OpenProcessToken.restype = wintypes.BOOL
        a.GetTokenInformation.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        a.GetTokenInformation.restype = wintypes.BOOL
        a.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        a.ConvertSidToStringSidW.restype = wintypes.BOOL
        a.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(wintypes.DWORD),
        ]
        a.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
        a.SetNamedSecurityInfoW.argtypes = [
            wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ]
        a.SetNamedSecurityInfoW.restype = wintypes.DWORD
        a.GetNamedSecurityInfoW.argtypes = [
            wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        a.GetNamedSecurityInfoW.restype = wintypes.DWORD
        a.GetSecurityDescriptorOwner.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.BOOL),
        ]
        a.GetSecurityDescriptorOwner.restype = wintypes.BOOL
        a.GetSecurityDescriptorDacl.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(wintypes.BOOL),
        ]
        a.GetSecurityDescriptorDacl.restype = wintypes.BOOL
        a.GetSecurityDescriptorControl.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(wintypes.WORD), ctypes.POINTER(wintypes.DWORD),
        ]
        a.GetSecurityDescriptorControl.restype = wintypes.BOOL
        a.GetAclInformation.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
        ]
        a.GetAclInformation.restype = wintypes.BOOL
        a.GetAce.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)]
        a.GetAce.restype = wintypes.BOOL
        a.EqualSid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        a.EqualSid.restype = wintypes.BOOL

    @staticmethod
    def _raise_winerror(operation: str, code: int | None = None) -> None:
        value = ctypes.get_last_error() if code is None else code
        raise PrivatePermissionError(f"{operation} failed with Windows error {value}")

    def _current_user_sid(self) -> tuple[str, ctypes.Array, ctypes.c_void_p]:
        token = wintypes.HANDLE()
        if not self.advapi32.OpenProcessToken(
            self.kernel32.GetCurrentProcess(), self.TOKEN_QUERY, ctypes.byref(token),
        ):
            self._raise_winerror("OpenProcessToken")
        try:
            size = wintypes.DWORD()
            self.advapi32.GetTokenInformation(token, self.TOKEN_USER_CLASS, None, 0, ctypes.byref(size))
            if ctypes.get_last_error() != self.ERROR_INSUFFICIENT_BUFFER or not size.value:
                self._raise_winerror("GetTokenInformation", ctypes.get_last_error())
            buffer = ctypes.create_string_buffer(size.value)
            if not self.advapi32.GetTokenInformation(
                token, self.TOKEN_USER_CLASS, buffer, size, ctypes.byref(size),
            ):
                self._raise_winerror("GetTokenInformation")
            user = ctypes.cast(buffer, ctypes.POINTER(_TOKEN_USER)).contents
            sid_pointer = ctypes.c_void_p(user.User.Sid)
            sid_text = self._sid_to_string(sid_pointer)
            return sid_text, buffer, sid_pointer
        finally:
            self.kernel32.CloseHandle(token)

    def _sid_to_string(self, sid: ctypes.c_void_p) -> str:
        result = ctypes.c_void_p()
        if not self.advapi32.ConvertSidToStringSidW(sid, ctypes.byref(result)):
            self._raise_winerror("ConvertSidToStringSidW")
        try:
            return ctypes.wstring_at(result.value)
        finally:
            self.kernel32.LocalFree(result)

    def _security_descriptor(self, sddl: str) -> ctypes.c_void_p:
        result = ctypes.c_void_p()
        if not self.advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl, 1, ctypes.byref(result), None,
        ):
            self._raise_winerror("ConvertStringSecurityDescriptorToSecurityDescriptorW")
        return result

    def protect(self, path: Path, *, directory: bool) -> None:
        full_path = str(path.absolute())
        user_sid, sid_buffer, user_sid_pointer = self._current_user_sid()
        # Keep the SID buffer alive while the descriptor is applied and checked.
        _ = sid_buffer
        inheritance = "OICI" if directory else ""
        aces = [f"(A;{inheritance};FA;;;{user_sid})"]
        if user_sid != self.SYSTEM_SID:
            aces.append(f"(A;{inheritance};FA;;;SY)")
        descriptor = self._security_descriptor(f"O:{user_sid}D:P{''.join(aces)}")
        try:
            owner = ctypes.c_void_p()
            dacl = ctypes.c_void_p()
            owner_defaulted = wintypes.BOOL()
            dacl_present, dacl_defaulted = wintypes.BOOL(), wintypes.BOOL()
            if not self.advapi32.GetSecurityDescriptorOwner(
                descriptor, ctypes.byref(owner), ctypes.byref(owner_defaulted),
            ):
                self._raise_winerror("GetSecurityDescriptorOwner")
            if not self.advapi32.GetSecurityDescriptorDacl(
                descriptor, ctypes.byref(dacl_present), ctypes.byref(dacl), ctypes.byref(dacl_defaulted),
            ) or not dacl_present.value or not dacl.value:
                self._raise_winerror("GetSecurityDescriptorDacl")
            information = (
                self.OWNER_SECURITY_INFORMATION
                | self.DACL_SECURITY_INFORMATION
                | self.PROTECTED_DACL_SECURITY_INFORMATION
            )
            result = self.advapi32.SetNamedSecurityInfoW(
                full_path, self.SE_FILE_OBJECT, information, owner, None, dacl, None,
            )
            if result:
                self._raise_winerror("SetNamedSecurityInfoW", result)
        finally:
            self.kernel32.LocalFree(descriptor)
        self._verify(full_path, directory=directory, user_sid=user_sid, user_sid_pointer=user_sid_pointer)

    def _verify(
        self,
        path: str,
        *,
        directory: bool,
        user_sid: str,
        user_sid_pointer: ctypes.c_void_p,
    ) -> None:
        owner, group, dacl, sacl, descriptor = (ctypes.c_void_p() for _ in range(5))
        result = self.advapi32.GetNamedSecurityInfoW(
            path, self.SE_FILE_OBJECT,
            self.OWNER_SECURITY_INFORMATION | self.DACL_SECURITY_INFORMATION,
            ctypes.byref(owner), ctypes.byref(group), ctypes.byref(dacl),
            ctypes.byref(sacl), ctypes.byref(descriptor),
        )
        if result:
            self._raise_winerror("GetNamedSecurityInfoW", result)
        try:
            actual_owner, dacl_pointer = ctypes.c_void_p(), ctypes.c_void_p()
            owner_defaulted, dacl_present, dacl_defaulted = wintypes.BOOL(), wintypes.BOOL(), wintypes.BOOL()
            control, revision = wintypes.WORD(), wintypes.DWORD()
            if not self.advapi32.GetSecurityDescriptorOwner(
                descriptor, ctypes.byref(actual_owner), ctypes.byref(owner_defaulted),
            ):
                self._raise_winerror("GetSecurityDescriptorOwner")
            if not self.advapi32.EqualSid(actual_owner, user_sid_pointer):
                raise PrivatePermissionError("The local credential path owner is not the current user")
            if not self.advapi32.GetSecurityDescriptorControl(
                descriptor, ctypes.byref(control), ctypes.byref(revision),
            ):
                self._raise_winerror("GetSecurityDescriptorControl")
            if not control.value & self.SE_DACL_PROTECTED:
                raise PrivatePermissionError("The local credential path inherits its DACL")
            if not self.advapi32.GetSecurityDescriptorDacl(
                descriptor, ctypes.byref(dacl_present), ctypes.byref(dacl_pointer), ctypes.byref(dacl_defaulted),
            ) or not dacl_present.value or not dacl_pointer.value:
                raise PrivatePermissionError("The local credential path has no private DACL")
            info = _ACL_SIZE_INFORMATION()
            if not self.advapi32.GetAclInformation(
                dacl_pointer, ctypes.byref(info), ctypes.sizeof(info), self.ACL_SIZE_INFORMATION_CLASS,
            ):
                self._raise_winerror("GetAclInformation")
            expected_sids = {user_sid, self.SYSTEM_SID}
            if info.AceCount != len(expected_sids):
                raise PrivatePermissionError("The local credential path DACL has unexpected entries")
            expected_flags = (
                self.OBJECT_INHERIT_ACE | self.CONTAINER_INHERIT_ACE if directory else 0
            )
            found = set()
            for index in range(info.AceCount):
                ace_pointer = ctypes.c_void_p()
                if not self.advapi32.GetAce(dacl_pointer, index, ctypes.byref(ace_pointer)):
                    self._raise_winerror("GetAce")
                header = ctypes.cast(ace_pointer, ctypes.POINTER(_ACE_HEADER)).contents
                if header.AceType != self.ACCESS_ALLOWED_ACE_TYPE or header.AceFlags != expected_flags:
                    raise PrivatePermissionError("The local credential path DACL contains an unexpected ACE")
                ace = ctypes.cast(ace_pointer, ctypes.POINTER(_ACCESS_ALLOWED_ACE)).contents
                if ace.Mask != self.FILE_ALL_ACCESS:
                    raise PrivatePermissionError("The local credential path DACL has unexpected access rights")
                sid_offset = ctypes.sizeof(_ACE_HEADER) + ctypes.sizeof(wintypes.DWORD)
                ace_sid = ctypes.c_void_p(ace_pointer.value + sid_offset)
                found.add(self._sid_to_string(ace_sid))
            if found != expected_sids:
                raise PrivatePermissionError("The local credential path DACL is not limited to this user and SYSTEM")
        finally:
            self.kernel32.LocalFree(descriptor)


__all__ = ["PrivatePermissionError", "protect_private_path"]

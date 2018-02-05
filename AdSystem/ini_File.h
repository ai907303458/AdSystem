#pragma once
#include <afxwin.h>
#include <iostream> 
#include <cstring> 
#include <stdio.h>
#include <windows.h>

using namespace std;

class Ini
{
public:
	Ini(char *ini_path)
	{
		path = new char[MAX_PATH];
		strcpy(path, ini_path);
	}
	bool setProfile(char *SectionName, char *keyName, char * key)
	{
		LPCWSTR a = Char2LPCWSTR(SectionName);
		LPCWSTR b = Char2LPCWSTR(keyName);
		LPCWSTR c = Char2LPCWSTR(key);
		cout <<"aaaaaaa "<<path << endl;
		return WritePrivateProfileString(Char2LPCWSTR(SectionName), Char2LPCWSTR(keyName), Char2LPCWSTR(key), Char2LPCWSTR(path));
	}
	char *getProfile(char *SectionName, char *keyName)
	{
		CString inBuf;
		GetPrivateProfileString(Char2LPCWSTR(SectionName), Char2LPCWSTR(keyName), TEXT("Error: Get profile failed"), inBuf.GetBuffer(MAX_PATH), MAX_PATH, Char2LPCWSTR(path));
		return CString2char(inBuf);
	}
private:
	LPCWSTR Char2LPCWSTR(char *s)
	{
		int dwLen = strlen(s) + 1;
		int nwLen = MultiByteToWideChar(CP_ACP, 0, s, dwLen, NULL, 0);//算出合适的长度
		LPWSTR lpszPath = new WCHAR[dwLen];
		MultiByteToWideChar(CP_ACP, 0, s, dwLen, lpszPath, nwLen);
		return lpszPath;
	}
	char * CString2char(CString s)
	{
		char *c = new char[MAX_PATH];
		DWORD dwNum = WideCharToMultiByte(CP_OEMCP, NULL, s, -1, NULL, NULL, 0, NULL);
		WideCharToMultiByte(CP_OEMCP, NULL, s, -1, c, dwNum, 0, NULL);
		return c;
	}
	char *path;
};
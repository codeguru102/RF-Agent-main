@echo off
setlocal EnableDelayedExpansion

REM Folder to scan. Use current folder by default.
set "ROOT=%cd%"

REM Change this to 0 when you want to really rename.
REM 1 = preview only, 0 = rename
set "TEST=1"

echo Scanning: "%ROOT%"
echo.

REM Rename files first
for /r "%ROOT%" %%F in (*candidate_00*) do (
    set "OLDNAME=%%~nxF"
    set "NEWNAME=!OLDNAME:candidate_00=candidate_00!"

    if not "!OLDNAME!"=="!NEWNAME!" (
        if exist "%%~dpF!NEWNAME!" (
            echo SKIP file exists: "%%~dpF!NEWNAME!"
        ) else (
            if "%TEST%"=="1" (
                echo FILE: "%%F"  --^>  "!NEWNAME!"
            ) else (
                ren "%%F" "!NEWNAME!"
                echo Renamed file: "%%F"  --^>  "!NEWNAME!"
            )
        )
    )
)

echo.
echo Renaming folders...

REM Rename folders deepest first
for /f "delims=" %%D in ('dir "%ROOT%\*candidate_00*" /ad /b /s ^| sort /R') do (
    set "OLDNAME=%%~nxD"
    set "NEWNAME=!OLDNAME:candidate_00=candidate_00!"

    if not "!OLDNAME!"=="!NEWNAME!" (
        if exist "%%~dpD!NEWNAME!" (
            echo SKIP folder exists: "%%~dpD!NEWNAME!"
        ) else (
            if "%TEST%"=="1" (
                echo FOLDER: "%%D"  --^>  "!NEWNAME!"
            ) else (
                ren "%%D" "!NEWNAME!"
                echo Renamed folder: "%%D"  --^>  "!NEWNAME!"
            )
        )
    )
)

echo.
echo Done.
pause
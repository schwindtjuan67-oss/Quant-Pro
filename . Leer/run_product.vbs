Set WshShell = CreateObject("WScript.Shell")

' 0 = ventana completamente oculta
WshShell.Run Chr(34) & "run_product.bat" & Chr(34), 0

Set WshShell = Nothing

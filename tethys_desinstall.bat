@echo off

echo Desinstalando Python e bibliotecas...
pip freeze > requirements.txt
pip uninstall -r requirements.txt -y
python -m venv --clear env
rmdir /s /q env
del requirements.txt
echo Python e bibliotecas desinstalados com sucesso.

echo Pressione qualquer tecla para sair.
pause > nul

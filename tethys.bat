@echo off

echo Instalando Python...
choco install python --yes
echo Python instalado com sucesso.

echo Instalando pip...
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
del get-pip.py
echo pip instalado com sucesso.

echo Configurando ambiente virtual...
python -m venv env
.\env\Scripts\activate.bat
echo Ambiente virtual configurado com sucesso.

echo Instalando bibliotecas...
pip install -r requirements.txt
echo Bibliotecas instaladas com sucesso.

echo Rodando script...
python main.py
echo Script finalizado.

echo Desativando o ambiente virtual...
deactivate
echo Ambiente virtual desativado.

echo Pressione qualquer tecla para sair.
pause > nul

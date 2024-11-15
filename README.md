How to Set Up and Run the Code 

For running the code for Task 4

For Problem 4 the files DQN_2_lay_targetN.ipynb and 
Q_learning_vs_SARSA.ipynb was programmed in Google Colab and copied into GitHub. The file PSO_VRPTW_alt3.ipynb was an initital flawed attempt 
for a solution for Problem 3 and should be disregarded as a part of the portfolio. 


For running the code for Task2 and Task3

1.Install Visual studio code:
https://code.visualstudio.com/download 
Install python 3.12.7:
https://www.python.org/downloads/release/python-3127/ 
(check mark: Add python.exe to PATH)
Make sure you have Visual Studio Code installed along with the extensions for Python and Jupyter Notebooks.
Make sure to have extensions in visual studio code:
I got:
-Python 
-python debugger 
-pylance 
-jupyter slide show 
-Jupyter Notebook Renderers  
-jupyter keymap 
-jupyter cell tags 
-Jupyter

2.Open terminal
Click Terminal top left of visual studio code -> click new terminal

3.Clone the Repository 
write in the terminal
git clone https://github.com/Frexander/Final_Portfolio.git 
This command will create a local copy of our repo on your machine. 

4.Going inside of the Project folder,write (you could write "cd Fin" and tab)
cd .\Final_Portfolio\ 
You will go to the inside of the project folder 

5.Install Dependencies 
pip install -r requirements.txt 
This will install all the required Python packages listed in requirements.txt. Important to have Python installed before running this command. 
How it looks: ...\Final_Portfolio> pip install -r requirements.txt 

6.Confirm the Packages  (Optional)
pip freeze 
just to confirm that all the necessary dependencies are installed. You should see a list of installed packages matching the content inside the requirements.txt. 
It can looked like this for me:
...\Final_Portfolio> pip freeze 
beautifulsoup4==4.12.3
certifi==2024.8.30
charset-normalizer==3.4.0
cloudpickle==3.1.0
contourpy==1.3.1
cycler==0.12.1
Farama-Notifications==0.0.4
filelock==3.16.1
fonttools==4.55.0
frozendict==2.4.6
fsspec==2024.10.0
gymnasium==1.0.0
html5lib==1.1
idna==3.10
Jinja2==3.1.4
kiwisolver==1.4.7
lxml==5.3.0
MarkupSafe==3.0.2
matplotlib==3.9.2
mpmath==1.3.0
multitasking==0.0.11
networkx==3.4.2
numpy==2.1.3
packaging==24.2
pandas==2.2.3
peewee==3.17.8
pillow==11.0.0
platformdirs==4.3.6
pyparsing==3.2.0
python-dateutil==2.9.0.post0
pytz==2024.2
requests==2.32.3
setuptools==75.5.0
six==1.16.0
soupsieve==2.6
sympy==1.13.1
torch==2.5.1
typing_extensions==4.12.2
tzdata==2024.2
urllib3==2.2.3
webencodings==0.5.1
yfinance==0.2.49


7.Open the Project in Visual Studio Code by writing in terminal
code . 
This command will open the project in Visual Studio Code.
it looks like:...\Final_Portfolio> code .


8. Click on Task2.ipynb or Task3.ipynb and click "Run All" (Jupyter Notebook extension)
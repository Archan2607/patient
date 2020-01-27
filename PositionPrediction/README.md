##Installation Requirements

The parent folder PositionPrediction contains a file named requirements.txt. It contains the packages that need to be installed for running the program.  
First of all, python needs to be installed in the system. The steps for installing python are given below for both Windows and Linux environments.  

Windows:  

1. Open a browser window and navigate to the Download page for Windows at python.org.  
2. Underneath the heading at the top that says Python Releases for Windows, click on the link for Python 3.7.0 Release.  
3. Scroll to the bottom and select either Windows x86-64 executable installer for 64-bit or Windows x86 executable installer for 32-bit.  

Linux:-  

1. Open the terminal and type the following commands:-  
    
    a. sudo add-apt-repository ppa:deadsnakes/ppa  
    b. sudo apt update  
    c. sudo apt install python3.7  
    
2. Install a venv to manage virtual environments by the following command  
    a. sudo apt-get install python3.7-venv
3. Create a virtual environment using the following command (It will create virtual environment in your home directory:-  
    a. python3.7 -m venv ~/<name_of_project>. (Default name used is “pos_pred”. If the name of the virtual environment is changed here then it needs to be changed in train.sh and classify.sh files as well.)  
4. Activate the virtual environment by the following command :-  
    a. source ~/<name_of_project>/bin/activate
5. Then install pip by running   
    a. sudo apt install python3.7-pip  
    
The below packages also need to be installed-  

1. pandas  
2. numpy  
3. sklearn  
4. matplotlib  

To install above packages go to the parent folder location and execute the below command in terminal.  
pip3.7 install -r requirements.txt in the parent directory.


## How to use Application?  

In the parent folder there are two bat files each for running the learning and classification pipeline for Windows and Linux environments respectively.  

Windows :-  

For training a model,  

1. keep the training data in data_train directory  
2. Double click on train.bat file. It will generate a model and required parameters file in the model folder.  

For classifying new data,  

1. Keep the new data in data_test directory  
2. Double click on classify.bat file. It will create a classified file which will contain classified positions of samples.  

Linux :-  

For training a model,  

1. Run chmod +x train.sh in the directory to make this file runnable. It needs to be run only once  
2. keep the training data in data_train directory  
3. Run using the command ./train.sh to run the learning pipeline. (Note please ensure if the name of created virtual environment is something other than the default value then it needs to be changed in this file as well)  

For classifying new data,

1. Run chmod +x classifys.sh in the directory to make this file runnable. It needs to be run only once
2. Keep the new data in data_test directory
3. Run using the command ./classify.sh to run the classification pipeline.  (Note please ensure if the name of created virtual environment is something other than the default value then it needs to be changed in this file as well)  


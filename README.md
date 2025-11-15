[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/R05VM8Rg)
# IIT-Madras-DA2401-Machine-Learning-Lab-End-Semester-Project

## 📌 Purpose of this Template

This repository is the **starter** for your End Semester Project submission in GitHub Classroom. You can implement your solution and push your work in this repository. Please free to edit this README.md file as per your requirements.

> **Scope (as per assignment brief):**
> Give an introduction to your repository here: Eg. This repository contains a complete implementation for ...

---

**Important Note:** 
1. TAs will evaluate using the `.py` file only.
2. All your reports, plots, visualizations, etc pertaining to your solution should be uploaded to this GitHub repository

---

## 📁 Repository Structure

* Describe your repository structure here. Explain about overall code organization.

I have uploaded three files which include Main Python file (main.py) to train the models on the data, and predict the results, algorithms.py file with implementation of all algorithms used in your system and a PDF report

## 📦 Installation & Dependencies

* Mention all the related instructions for installation of related packages for running your code here.

import numpy as np
import pandas as pd
import time
from collections import Counter

## ▶️ Running the Code

All experiments should be runnable from the command line **and** reproducible in the notebook.

### A. Command-line (recommended for grading)

* Mention the instructions to run you .py files.
  
In the main.py file, I have used the MNIST_train.csv file to train the model and I am validating using MNIST_validation.csv. So, for testing the model using the test dataset, you can replace the path of the MNIST_validation.csv with the path of the test dataset. At the end of the code, the accuracy of the model will be printed and all the predictions will be stored in a variable 'final_predictions'.

In the algorithms.py file, I have trained many models and I have chosen the best one for the main.py.

## You can further add your own sections/titles along with corresponding contents here:

---

## 🧾 Authors

**<Prabhav Gupta, Roll No. DA24B018>**, IIT Madras (2025–26)


## Best Practices:
* Keep commits with meaningful messages.
* Please do not write all code on your local machine and push everything to GitHub on the last day. The commits in GitHub should reflect how the code has evolved during the course of the assignment.
* Collaborations and discussions with other students is strictly prohibited.
* Code should be modularized and well-commented.


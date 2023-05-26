# Path Planning with D* Lite
## Description
This GitHub repository tracks a project to explore path-planning with D*-Lite, an algorithm first introduced in the paper
["D* Lite](http://idm-lab.org/bib/abstracts/papers/aaai02b.pdf) by Sven Koenig and Maxim
Likachev. 

The project is artificially simulating terrain and implementing A* and D* Lite path planning.
It aims to compare the performance of the algorithms when planning in outdoor environments 
when initially planning on an estimate of the terrain but then updated as you add information about the terrain. 

## Table of Contents
- [Installation](#installation)
- [Usage](#Usage)

## Installation
This project uses the Conda interpreter. In your Conda command prompt, navigate to your local repository. 
You can install the conda environment using the command 
`conda env create -f environment.yml`. 

## Usage 
In the Conda command prompt, activate your installed environment. Run the command `jupyter notebook` in the command 
prompt. Open the notebook titled `DSL Sim Notebook` and run through the notebook. 
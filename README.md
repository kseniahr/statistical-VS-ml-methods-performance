# Statistical and ML models performance evaluation
This artifact defines the most robust model for long-term predictions by conducting a simulation study and comparing the results of the experiments.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install -r /path/to/requirements.txt
```

## Download repository
Open a terminal and navigate it to the folder where you want to save a project

```bash
cd /path/to/folder
```

Clone this repository to your local environment with HTTPS:
```bash
git clone https://github.com/kseniahr/statistical-VS-ml-methods-performance.git
```
or with SSH:
```bash
git clone git@github.com:kseniahr/statistical-VS-ml-methods-performance.git
```

## Simulation set-up
Open simulation-model/Main.py with a text editor (I am using Atom.io) and define the settings for a simulation. The default settings are:
```
# Define data generation settings:

N = 1000  # Observations in the population
b = 3     # Number of predictor variables
k = 10    # Number of samples
n = 100   # Observations in each sample
y = 2019  # Starting year
t = 10    # Number of years
```
## Run simulation
Navigate to a folder ./simulation-model:


```bash
cd /simulation-model
```
and type in 

```bash
python Main.py
```
You will have to select which function will describe the data. The alternatives proposed in this study are a) linear regression function:

![equation](https://latex.codecogs.com/gif.latex?Y%20%3D%20%5Cbeta_%7B%7D%200%20&plus;%5Cbeta_%7B%7D%201%20X_%7B%7D%201&plus;%5Cbeta_%7B%7D%202%20X_%7B%7D%202%20&plus;%C3%82%C2%B7%C3%82%C2%B7%C3%82%C2%B7&plus;%5Cbeta%20iXi%20&plus;%20%5Cvarepsilon%251%251) 

or b) polynomial regression function:

![equation](https://latex.codecogs.com/gif.latex?Y%20%3D%20%5Cbeta_%7B%7D%200%20&plus;%5Cbeta_%7B%7D%201%20X_%7B%7D%201&plus;%5Cbeta_%7B%7D%202%20X_%7B%7D%202%5E%7B2%7D%20&plus;%20.%20.%20.%20&plus;%5Cbeta%20iXn%5E%7Bn%7D%20&plus;%20%5Cvarepsilon%251%251)

Then, you will have to choose which of the following studies do you want to run:

a) study1 - evaluates the overtime performance if the distribution of input variables changes (X1,X2,..., Xn)

b) study2 - evaluates the overtime performance if relationship between input variables changes (B1,B2,...,Bi)

c) study3 - evaluates the overtime performance if mean of the dependent variable changes

d) study4 - evaluates the overtime performance of all 3 studies combined

After the script Main.py is executed, the figures of MSE performance will be saved to the folder ./simulation-model/plots/


## Contributing
Pull requests are welcome on a separatelly created branch 'development'. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update simulation models as appropriate.


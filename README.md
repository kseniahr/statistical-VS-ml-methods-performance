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

n_X           = 3           # Number of predictor variables (min = 3)
start_year    = 2019        # Starting year
n_years       = 10          # Number of years
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

![equation](https://latex.codecogs.com/gif.latex?Y%20%3D%20%5Cbeta_%7B%7D%200%20&plus;%5Cbeta_%7B%7D%201%20X_%7B%7D%201&plus;%5Cbeta_%7B%7D%202%20X_%7B%7D%202%20&plus;...&plus;%5Cbeta%20iXn%20&plus;%20%5Cvarepsilon%251%251%251) 

or b) polynomial regression function:

![equation](https://latex.codecogs.com/gif.latex?Y%20%3D%20%5Cbeta_%7B%7D%200%20&plus;%5Cbeta_%7B%7D%201%20X_%7B%7D%201&plus;%5Cbeta_%7B%7D%202%20X_%7B%7D%202%5E%7B2%7D%20&plus;%20.%20.%20.%20&plus;%5Cbeta%20iXn%5E%7Bn%7D%20&plus;%20%5Cvarepsilon%251%251)

Then, you will have to select which variables should be used to generate a dataset. The alternatives proposed in this study are 

a) continuous

b) hybrid (both continuous and binary variables)

Then, you will be asked about the population size, sample size and number of samples.

Then, you will have to choose which of the following experiments do you want to run:

a) 1 - evaluates the overtime performance if the distribution of input variables changes (X1,X2,..., Xn)

b) 2 - evaluates the overtime performance if relationship between input variables changes (B1,B2,...,Bi)

c) 3 - evaluates the overtime performance if mean of the dependent variable changes

d) 4 - evaluates the overtime performance of all 3 studies combined

After the script Main.py is executed, the figures of MSE performance will be saved to the folder ./simulation-model/plots/


## Contributing
Pull requests are welcome on a separatelly created branch 'development'. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update simulation models as appropriate.


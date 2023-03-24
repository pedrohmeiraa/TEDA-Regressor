# TEDA Regressor: *An Online Processing of Vehicular Data on the Edge Through an Unsupervised TinyML Regression Technique*


## ✍🏾Authors:  Pedro Andrade, Ivanovitch Silva, Marianne Diniz, Thommas Flores, Daniel G. Costa, and Eduardo Soares.


# 1. Abstract/Overview
The Internet of Things (IoT) has made it possible to include everyday objects in a connected network, allowing them to intelligently process data and respond to their environment. Thus, it is expected that those objects will gain an intelligent understanding of their environment and be able to process data more efficiently than before. Particularly, such edge computing paradigm has allowed the execution of inference methods on resource-constrained devices such as microcontrollers, significantly changing the way IoT applications have evolved in recent years [[1]](https://www.mdpi.com/1424-8220/22/10/3838). However, although this scenario has supported the development of Tiny Machine Learning (TinyML) approaches on such devices, there are still some challenges that require further investigation when optimizing data streaming on the edge  [[2]](https://www.researchgate.net/publication/301411485_Anomaly_detection_based_on_eccentricity_analysis). 

Therefore, **in the article associated with this repository** [[3]](https://dl.acm.org/journal/tecs) proposes a *new unsupervised TinyML regression technique* based on the typicality and eccentricity of the samples to be processed. Moreover, the proposed technique also exploits a Recursive Least Squares (RLS) filter approach. Combining all these features, the proposed method uses similarities between samples to identify patterns when processing data streams, predicting outcomes based on these patterns. The results obtained through the extensive experimentation utilizing vehicular data streams were highly encouraging.

The proposed algorithm was meticulously compared with the RLS algorithm and Convolutional Neural Networks (CNN). It exhibited significantly superior performance, with mean squared errors that were 4.68 and 12.02 times lower, respectively, compared to the aforementioned techniques.
  
For a better didactic exposition, the results will be presented in 3 notebooks:

 1. :notebook:TEDA Regressor
 2. :orange_book: RLS - Recursive Least Squares
 3. :green_book: CNN - Convolutional Neural Networks

# 2. Environment Setup
First, start by cloning the repository:
```bash
git clone https://github.com/pedrohmeiraa/TEDA-Regressor
```

We also have cloned the `Padasip` repository:
```bash
git clone https://github.com/matousc89/padasip
```
It is possible to install using `pip`:
```bash
!pip3 install padasip
```
- The **Padasip** (*Python Adaptive Signal Processing*) is a library designed to simplify adaptive signal processing tasks within Python (filtering, prediction, reconstruction, classification). More information [here](https://matousc89.github.io/padasip/).  :twisted_rightwards_arrows:


Now, we are going to install the  `WandB`: 💻

```bash
!pip3 install wandb -qU
```
 - If you want to know more about software package **WandB**, click [here](https://wandb.ai/site).  :bar_chart:
 
You also have to install the `CodeCarbon`, running the following command:

```bash
!pip3 install codecarbon
```
 - If you want to know more about software package **CodeCarbon**, click [here](https://codecarbon.io/).  :leaves:

***

# 3. Steps for Definition of Aforementioned Methodology


For each of the algorithms, their respective notebook will present the following steps:
  3.1 `Installing modules`
  3.2 ``Data Acquisition``;
  3.2 ``Sweep Variables Definition``;
  3.3 ``WandB Sweep``;
  3.4 ``Data Analysis``.

### 3.1) Installing modules

First, he are going to show the main modules and dependencies to work:

```bash
## Padasip library
import  padasip  as  pa
from  padasip.filters.base_filter  import AdaptiveFilter

##WandB
import  wandb

##CodeCarbon
from  codecarbon  import  EmissionsTracker
```
And:
```bash
from  TedaRegressor  import  DataCloud, TEDARegressor
```

### 3.2) Data Acquisition

The updated raw data could be collected by the Jupyter Notebook or Python Script, that are contained into the folder ``TEDARegressor``.

If you choose to include the updated data, it is necessary to track and version the input files. So, you need to run the following commands:

```bash
git clone https://github.com/pedrohmeiraa/TEDA-Regressor/full_data.xlsx
```

If you want to use the version used, the raw data are available on  [here](https://github.com/pedrohmeiraa/TEDA-Regressor/raw/master/full_data.xlsx). Also, they are contained into the folder ``/full_data.xlsx``.

As the file is in ``.xlsx`` format, data acquisition can be done in a ``DataFrame`` of the Pandas tool:

```bash
df = pd.read_excel("full_data.xlsx")
```

### 3.3) Sweep variables definition

The first thing we need to define is the `method` for choosing new parameter values.
We provide the following search `methods`:
* **`grid` Search** – Iterate over every combination of hyperparameter values. Very effective, but can be computationally costly. **We choose the `grid` method.**
* **`random` Search** – Select each new combination at random according to provided `distribution`s. Surprisingly effective!
* **`bayesian` Search** – Create a probabilistic model of metric score as a function of the hyperparameters, and choose parameters with high probability of improving the metric. Works well for small numbers of continuous parameters but scales poorly.

We need to know its `name`, so we can find it in the model outputs and we need to know whether your `goal` is to `minimize` it (*e.g.* if it's the **mean squared error**) or to `maximize` it (*e.g.* if it's the **accuracy**).

```bash
metric_dict = {
'name': 'mse_TEDA',
'goal': 'minimize'
}
```

To try out new values of the hyperparameters, you need to define what those  `parameters`  are. We have to give the  `parameter`  a name and specify a list of legal  `values`  of the parameter:

#### 

**TEDA Regressor**

```bash
parameters_dict = {
#TEDA Regressor
	'Window': {
		'values': [2, 3, 4, 5, 6]
	},

	'Factor': {
		'values': [0.000001, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
	},

	'Threshold': {
		'values': [1.5, 1.75, 2, 2.25, 3]
	},

	'Init': {
		'values': ["relu", "tanh1", "tanh2", "zero"]
	},
}
```

**RLS**
```bash
parameters_dict = {
#RLS
	'Window': {
		'values': [2,3,4,5,6]
	},

	'Factor': {
		'values': [0.000001, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
	},
}
```
**CNN**
```bash
parameters_dict = {
#CNN
	'Window': {
		'values': [2,3,4,5,6]
	},

	'ActList': {
		'values': ['relu', 'linear']
	},

	'OptList': {
		'values': ['adam', 'sgd', 'rmsprop']
	},
}
```

#### Finally:
```bash
sweep_config = {
	"method": "grid",
	"metric": metric_dict,
	"parameters": parameters_dict,
}

sweep_id = wandb.sweep(sweep_config, project="TEDARegressor")
```

### 3.4) WandB sweep
 
 Now, we have to define our training procedure, before we can actually execute the sweep [[WandB Docs]](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb#scrollTo=IB1LeARB8Ccv):
In the functions below, we define the algorithm (TEDA Regressor, RLS or CNN), and add the following `wandb` tools to log model metrics, visualize performance and output and track our experiments 

* [**`wandb.init()`**](https://docs.wandb.com/library/init) – Initialize a new W&B Run. Each Run is a single execution of the training function.

* [**`wandb.config`**](https://docs.wandb.com/library/config) – Save all your hyperparameters in a configuration object so they can be logged. Read more about how to use `wandb.config`  [here](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-config/Configs_in_W%26B.ipynb).

* [**`wandb.log()`**](https://docs.wandb.com/library/log) – log model behavior to W&B. Here, we just log the performance; see [this Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb) for all the other rich media that can be logged with `wandb.log`.
 
#### 3.4.1 Defining our training procedure:

**TEDA Regressor**
```bash
def  train():
	with  wandb.init() as  run:
		# create codecarbon tracker
		# codecarbon is too much verbose, change the log level for more info
		tracker = EmissionsTracker(log_level="critical")
		tracker.start()

		X_train, y_train, X_test, y_test = treating_dataset(name_series = feature, window=wandb.config.Window, N_splits = 5)
		tedaRegressor=TEDARegressor(m=wandb.config.Window, mu=wandb.config.Factor, threshold=wandb.config.Threshold, activation_function=wandb.config.Init)

		for  t  in  X_test:
			#TEDARegressor
			tedaRegressor.run(np.array(t))
			# get co2 emissions from tracker
			# "CO2 emission (in Kg)"
			emissions = tracker.stop() #CONFERIR SE É AQUI MESMO

			# MSE TEDARegressor
			mse_TEDA = mean_squared_error(y_test[1:-1], TEDARegressor.Ypred)
			MSE_TEDA.append(mse_TEDA)
			  
			run.summary['mse_TEDA'] = mse_TEDA
			wandb.log({"mse_TEDA": mse_TEDA})

			# energy unit is kWh
			run.summary["Energy_Consumed"] = tracker.final_emissions_data.energy_consumed
			run.summary["Energy_RAM"] = tracker.final_emissions_data.ram_energy
			run.summary["Energy_GPU"] = tracker.final_emissions_data.gpu_energy
			run.summary["Energy_CPU"] = tracker.final_emissions_data.cpu_energy
			# kg
			run.summary["CO2_Emissions"] = tracker.final_emissions_data.emissions
```
**RLS**
```bash
def  train():
	with  wandb.init() as  run:
		# create codecarbon tracker
		# codecarbon is too much verbose, change the log level for more info
		tracker = EmissionsTracker(log_level="critical")
		tracker.start()

		X_train, y_train, X_test, y_test = treating_dataset(name_series = sum_series, window=wandb.config.Window, N_splits = 5)

		#RLS
		fRLS = pa.filters.FilterRLS(wandb.config.Window, mu=wandb.config.Factor, w="zeros")
		Y_RLS = []

		#X = X.T
		t_ant = X_test[0]
		count=0

		for  t  in  X_test:
		#RLS
			y = fRLS.predict(t)
			fRLS.adapt(t[-1], t_ant)
			t_ant = t
			Y_RLS.append(y)
			count+=1

		# get co2 emissions from tracker
		# "CO2 emission (in Kg)"
		emissions = tracker.stop() #CONFERIR SE É AQUI MESMO

		#RLS
		mse_RLS = mean_squared_error(y_test[1:-1], Y_RLS[2:])
		MSE_RLS.append(mse_RLS)
		run.summary['mse_RLS'] = mse_RLS
		wandb.log({"mse_RLS": mse_RLS})

		# energy unit is kWh
		run.summary["Energy_Consumed"] = tracker.final_emissions_data.energy_consumed
		run.summary["Energy_RAM"] = tracker.final_emissions_data.ram_energy
		run.summary["Energy_GPU"] = tracker.final_emissions_data.gpu_energy
		run.summary["Energy_CPU"] = tracker.final_emissions_data.cpu_energy

		# kg
		run.summary["CO2_Emissions"] = tracker.final_emissions_data.emissions
```

**CNN**
```bash
def  train():
	with  wandb.init() as  run:
		# create codecarbon tracker
		# codecarbon is too much verbose, change the log level for more info
		tracker = EmissionsTracker(log_level="critical")
		tracker.start()

		X_train, y_train, X_test, y_test = treating_dataset(name_series = sum_series, window=wandb.config.Window, N_splits = 5)

		# define model
		cnn_model = Sequential()
		cnn_model.add(Conv1D(filters=64, kernel_size=2, activation=wandb.config.ActList, input_shape=(wandb.config.Window, 1), padding='same'))
		cnn_model.add(MaxPooling1D(pool_size=2))
		cnn_model.add(Flatten())
		cnn_model.add(Dense(50, activation=wandb.config.ActList))
		cnn_model.add(Dense(1))

		cnn_model.compile(optimizer=wandb.config.OptList, loss='mse')

		# fit model
		cnn_model.fit(X_train, y_train, epochs=200, verbose=0)

		# get co2 emissions from tracker
		# "CO2 emission (in Kg)"
		emissions = tracker.stop() #CONFERIR SE É AQUI MESMO

		# to predict
		cnn_model.predict(X_test)

		#model predict
		y_predict_cnn = cnn_model.predict(X_test)

		# Calculating mse:
		mse_cnn = mean_squared_error(y_test, y_predict_cnn)
		MSE_CNN.append(mse_cnn)

		run.summary['mse_cnn'] = mse_cnn
		wandb.log({"mse_cnn": mse_cnn})

		# energy unit is kWh
		run.summary["Energy_Consumed"] = tracker.final_emissions_data.energy_consumed
		run.summary["Energy_RAM"] = tracker.final_emissions_data.ram_energy
		run.summary["Energy_GPU"] = tracker.final_emissions_data.gpu_energy
		run.summary["Energy_CPU"] = tracker.final_emissions_data.cpu_energy

		# kg
		run.summary["CO2_Emissions"] = tracker.final_emissions_data.emissions
```

#### So, now we are ready to start sweeping! 🧹🧹🧹
To get going on configuration, the `wandb.agent` needs to know:
1. which Sweep it's a part of (`sweep_id`)
2. which function it's supposed to run (here, `train`)

```bash
wandb.agent(sweep_id, train)
```

### 3.5) Data Analysis  
Here, we can visualize the Sweep Results 👀. We have use the **Parallel Coordinates Plot 🔀** to map hyperparameter values to model metrics.
We've used to see the combinations of hyperparameters that led to the **best model performance (minor MSE) 📊** and the **CodeCarbon results :leaves:**:

#### 3.5.1 TEDA Regressor sweep:

![TEDA Regressor Sweep.](https://github.com/pedrohmeiraa/TEDA-Regressor/blob/master/Figures/TEDA_sweep.png?raw=true)

#### 3.5.2 RLS sweep:

![RLS Sweep.](https://github.com/pedrohmeiraa/TEDA-Regressor/blob/master/Figures/RLS_sweep.png?raw=true)

#### 3.5.3 CNN sweep:

![CNN Sweep.](https://github.com/pedrohmeiraa/TEDA-Regressor/blob/master/Figures/CNN_sweep.png?raw=true)

### The sweeps used in the article are public and can be accessed at:

[1. TEDA Regressor Sweep]();
[2. RLS Sweep]() and;
[3. CNN Sweep]();

# 4. Citations


### How does it cite?

- **Andrade, P.**; Silva, I.; Silva, M.; Flores, T.; Costa, D.G. Soares, E.; _Online Processing of Vehicular Data on the Edge Through an Unsupervised TinyML Regression Technique_. ACM TECS 2023. ![GitHub](https://img.shields.io/badge/DOI-in%20submission%20process-blue)

### How does the article download?

- You can download the article from this [link](https://dl.acm.org/journal/tecs). ![Open in PDF](https://img.shields.io/badge/-PDF-EC1C24?style=flat-square&logo=adobeacrobatreader)

### How to download the data?

- You can download the data in this repository or clicking [here](https://github.com/pedrohmeiraa/TEDA-Regressor/raw/master/full_data.xlsx). 💹

# 5. References

 [[1]](https://www.mdpi.com/1424-8220/22/10/3838) :books: **Andrade, P.**; Silva, I.; Silva, M.; Flores, T.; Cassiano, J.; Costa, D.G. *A TinyML Soft-Sensor Approach for Low-Cost Detection and Monitoring of Vehicular Emissions*. SENSORS 2022, 22, 3838.  ![GitHub](https://img.shields.io/badge/DOI-10.3390%2Fs22103838-green)

[[2]](https://www.mdpi.com/1424-8220/21/12/4153) :books: Signoretti, G. ; Silva, M. ; **Andrade, P.**; Silva, I. ; Sisinni, E. ; Ferrari, P.; *An Evolving TinyML Compression Algorithm for IoT Environments Based on Data Eccentricity*. SENSORS, v. 21, p. 4153, 2021. ![GitHub](https://img.shields.io/badge/DOI-10.3390%2Fs21124153-green)

[[3]](https://dl.acm.org/journal/tecs) :books: **Andrade, P.**; Silva, I.; Silva, M.; Flores, T.; Costa, D.G. Soares, E.; _Online Processing of Vehicular Data on the Edge Through an Unsupervised TinyML Regression Technique_. ACM TECS 2023. ![GitHub](https://img.shields.io/badge/DOI-in%20submission%20process-blue)


# Deep image prior with total variation regularization to reconstruct Electrical Impedance Tomography images from limited data

## Brief description of our algorithm in the context of Kuopio Tomography Challenge 2023 (KTC2023) [[1]](#1), [[2]](#2)

Our algorithm is based on the Deep Image Prior (DIP) [[3]](#3) with Total Variation (TV) regularization. This is our second submission to the KTC2023.

* DIP input: Difference voltage data $\Delta V$ (Full or limited measurements)
* DIP output: Reconstructed conductivity change $\Delta \sigma$

Regarding DIP, the generative deep convolutional neural network is a parametric function $f_{\theta}(z)$, where the generator weights $θ$ are randomly initialized and $z$ is an image that represents the tank with water.  

During the traning phase, the weights are adjusted to map $f_{\theta}(z)$ to the conductivities $\Delta \sigma$. The loss function is given by:

$\hat{\theta}$ = $\arg\underset{\theta}{\min}$ $E (J f_{\theta}(z), \Delta V) + TV(\Delta V)$,  

where $\hat{\theta}$ are the weights of the generator network $f$ after fitting, $J$ is the jacobian of the forward EIT problem, $TV$ is the total variation regularizer and $E$ is the loss function.  

After this, the reconstructed image is given by $\Delta\hat{\sigma} = f_{\hat{\theta}}(z)$.

Then, $\Delta \sigma$ is interpolated to a regular 256 x 256 grid and segmented into water, conductive inclusions, and resistive inclusions. 



## Authors
* Leonardo Alves Ferreira¹* - leonardo.alves@ufabc.edu.br
* Roberto Gutierrez Beraldo¹ - roberto.gutierrez@ufabc.edu.br
* Fernando Silva de Moura² - fernando.moura@ufabc.edu.br
* André Kazuo Takahata¹ - andre.t@ufabc.edu.br
* Ricardo Suyama¹ - ricardo.suyama@ufabc.edu.br
  
*Corresponding author

¹Federal University of ABC - [UFABC](https://www.ufabc.edu.br/) - Campus Santo André - Avenida dos Estados, 5001 - Bairro Bangu, Santo André - CEP: 09280-560 (Brazil)

²Federal University of ABC - [UFABC](https://www.ufabc.edu.br/) - Campus São Bernardo - Alameda da Universidade, s/nº - Bairro Anchieta, São Bernardo do Campo - CEP: 09606-405 (Brazil)

## Proposed method installation

### Method installation and requirements
* The Python codes are available in this repository, see main.py and the /utils folder.
* To clone this repository:
  
     git clone https://github.com/robert-abc/KTC2023-ABC

     cd KTC2023-ABC

### Prerequisites
* In the following table, there is a small list of the main packages we used (with "import").

| Package | Version |
| ------------- | ------------- |
| Python | 3.10.11 | 
| Numpy | 1.23.5 | 
| Matplotlib | 3.7.1 | 
| Scipy | 1.11.3 | 
| Pillow | 9.4.0 | 
| Torch | 2.1.0+cu118 | 
| Torchmetrics | 1.2.0 | 

### External codes

* We used part of the codes provided in the challenge to implement the EIT forward model.
* We also need to mention that we adapted functions from the original DIP article [[4]](#4). Available at https://github.com/DmitryUlyanov/deep-image-prior/, under Apache License 2.0. The particular requisites are shown here: https://github.com/DmitryUlyanov/deep-image-prior/blob/master/README.md

## Usage instructions and example: Running with a callable function from the command line

By the rules of the KTC2023, it was expected a main routine with three arguments: 
* *Your main routine must require three input arguments:*
1. *(string) Folder where the input files are located (.mat files)*
1. *(string) Folder where the output files must be stored*
1. *(int) (int) Group category number (Values between 1 and 7)*
* *Python: The main function must be a callable function from the command line. 

After the setup, it is possible to run our code following these rules. Considering the difficulty group 7: 
* python main.py 'input' 'example/output' 7

See, for instance, the Section "Generating results" from the example notebook [Here](/notebook_example.ipynb).

## References

<a id="1">[1]</a> 
M. Räsänen et al.
“Kuopio Tomography Challenge 2023”. [Online]. Available at: https://www.fips.fi/KTC2023.php.

<a id="2">[2]</a> 
M. Räsänen et al. 
“Kuopio Tomography Challenge 2023 open electrical impedance tomographic dataset (KTC 2023)”. Available at: https://zenodo.org/records/8252370.

<a id="1">[3]</a> 
D. Ulyanov, A. Vedaldi, and V. Lempitsky.
“Deep image prior”. International Journal of Computer Vision, vol. 128, no. 7, pp.1867–1888 (2020). [Online]. Available at: https://doi.org/10.1007/s11263-020-01303-4

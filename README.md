<a name="readme-top"></a>
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://medik.tf.itb.ac.id/profil/">
    <img src="images/lab-logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Double Stent Deployment Simulation for Aneurysm Therapy</h3>

  <p align="center">
    Simulation of deployment of double stent using Fast Virtual Stenting Algorithm.
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

This project was initialized by <a href="https://sites.google.com/view/narendkurnia/home?authuser=0">Narendra Kurnia Putra Ph.D.</a> and <a href="https://www.linkedin.com/in/bonfilio-nainggolan-12508415a/">Bonfilio Nainggolan</a> as Bonfilio's undergraduate thesis project in 2021. Back then, the purpose of this project was to analyse the effect of flow diverter stent therapy using single stent layer and the result was published on <a href="https://ieeexplore.ieee.org/document/9624474">here</a>.

For further improvement, this project is extended to analyze the effect of double stent therapy using CFD.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

<br/>

### Installation
To get a local copy up and running follow these simple example steps.


1. Install <a href="https://www.anaconda.com/">Anaconda</a> 
2. Clone the repo
   ```cmd
   git clone https://github.com/rafisudrajat/aneurysm-stent-sym.git
   ```
3. Create conda virtual environment and activate it
   ```cmd
   conda create -n [env-name] python=3.9
   conda activate [env-name]
   ```
4. Install python package in `requirements.txt`
   ```cmd
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### How to run

1. Open the `runSym.cmd` and select the simulation directory by changing the dir param variable
  ```cmd
   set "dir_param=experiment\[experiment_dir]"
  ```
1. Set the simulation setting by changing the appSettings.json file inside the [experiment_dir]
2. Run the simulation using `runSym.cmd`
  ```cmd
   runSym.cmd
  ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>


[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/muhammad-rafi-sudrajat/
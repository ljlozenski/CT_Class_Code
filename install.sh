
#conda create --name ct_class python=3.12
#conda activate ct_class
#conda install -c https://software.repos.intel.com/python/conda -c conda-forge -c ccpi cil=25.0.0
#conda install jupyter
#conda install conda-forge::astra-toolbox=2.1=cuda*
#conda install ccpi::tigre=2.6
#conda install ccpi::tomophantom=2.0.0
#pip install torch torchvision
##jupyter notebook --port=44000 --ip=$HOSTNAME --no-browser

mkdir Datasets/
curl -L -o Datasets/chest-ctscan-images.zip  https://www.kaggle.com/api/v1/datasets/download/mohamedhanyyy/chest-ctscan-images
unzip Datasets/chest-ctscan-images.zip -d Datasets/chest_data

curl -L -o ct-to-mri-cgan.zip https://www.kaggle.com/api/v1/datasets/download/darren2020/ct-to-mri-cgan
unzip Datasets/ct-to-mri-cgan.zip -d Datasets/brain_data

mkdir Figures

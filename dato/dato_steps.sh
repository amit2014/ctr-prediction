# navigate to ctr-prediction folder
cd ctr-prediction

# create a new virtual environment with Python 2.7.x
virtualenv -p /usr/bin/python2.7 dato-env

# activate the virtual environment
source dato-env/bin/activate

# ensure pip is updated to the latest version
pip install --upgrade pip

# install your licensed copy of GraphLab Create
pip install --upgrade --no-cache-dir https://get.dato.com/GraphLab-Create/1.8.5/your registered email address here/your product key here/GraphLab-Create-License.tar.gz

# when finished, deactivate the virtual environment
deactivate

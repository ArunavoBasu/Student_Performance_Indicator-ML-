### With "setup.py" the ML application can be used as a package, which we can use in different applications as well

from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        ### To ignore "-e ." while taking the inputs from requirements.txt file
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
name='Student_Performance_Indicator',
version='0.0.1',
author='BASU',
author_email='basu.arunavo@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')

)
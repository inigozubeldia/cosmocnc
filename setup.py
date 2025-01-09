from setuptools import setup


setup(
    name="cosmocnc",
    version="1.0",
    description="Python package for fast cluster number count likelihood computation",
    zip_safe=False,
    packages=["cosmocnc"],
    author = 'Inigo Zubeldia and Boris Bolliet',
    author_email = 'inigo.zubeldia@ast.cam.ac.uk',
    url = 'https://github.com/inigozubeldia/cosmocnc',
    download_url = 'https://github.com/inigozubeldia/cosmocnc',
    package_data={
        # "specdist": ["data/*txt"],
        #"data/ct_database/case_1_040520/*txt"]#,
    },


)

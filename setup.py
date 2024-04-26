from setuptools import setup




setup(
    name="cosmocnc",
    version="0.1",
    description="Python package for fast cnc",
    zip_safe=False,
    packages=["cosmocnc"],
    author = 'Inigo Zubeldia',
    author_email = 'inigo.zubeldia@ast.cam.ac.uk',
    url = 'https://github.com/inigozubeldia/cnc',
    download_url = 'https://github.com/inigozubeldia/cnc',
    package_data={
        # "specdist": ["data/*txt"],
        #"data/ct_database/case_1_040520/*txt"]#,
    },


)
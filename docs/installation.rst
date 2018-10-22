.. _installation:

Installation
============

For Users
+++++++++

``orbitize`` is registered on ``pip``, and works in Python 2.7 or 3.6.
To install ``orbitize``, first make sure you have the latest versions
of ``numpy`` and ``cython`` installed. With ``pip``, you can do this with
the command:

.. code-block:: bash
	
	$ pip install numpy cython --upgrade

Next, install ``orbitize``:

.. code-block:: bash
	
	$ pip install orbitize

If you are having difficulties installing orbitize with cython, you can
disable compilation of the C Kepler module with the following command:

.. code-block:: bash
	
	$ pip install orbitize --install-option="--disable-cython"

We recommend installing and running ``orbitize`` in a ``conda`` virtual
environment. Install ``anaconda`` or ``miniconda`` 
`here <https://conda.io/miniconda.html>`_, then see instructions 
`here <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_
to learn more about ``conda`` virtual environments.

For Developers
++++++++++++++

``orbitize`` is actively being developed. The following method for 
installing ``orbitize`` will allow you to use it and make changes to it. 
After cloning the Git repository, run the setup file in the top level 
of the repo:

.. code-block:: bash
	
	$ python setup.py develop

Issues?
+++++++

If you run into any issues installing ``orbitize``, please create an issue on GitHub.



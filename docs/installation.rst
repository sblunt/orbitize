.. _installation:

Installation
============

For Users
+++++++++

Parts of ``orbitize`` are written in C, so you'll need ``gcc`` (a C compiler) to install properly.
Most Linux and Windows computers come with ``gcc`` built in, but Mac computers don't. If you
haven't before, you'll need to download Xcode command line tools. There are several
helpful guides online that teach you how to do this. Let us know if you have trouble! 

``orbitize`` is registered on ``pip``, and works in Python>3.6.
To install ``orbitize``, first make sure you have the latest versions
of ``numpy`` and ``cython`` installed. With ``pip``, you can do this with
the command:

.. code-block:: bash
	
	$ pip install numpy cython --upgrade

Next, install ``orbitize``:

.. code-block:: bash
	
	$ pip install orbitize

We recommend installing and running ``orbitize`` in a ``conda`` virtual
environment. Install ``anaconda`` or ``miniconda`` 
`here <https://conda.io/miniconda.html>`_, then see instructions 
`here <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_
to learn more about ``conda`` virtual environments.

For Windows Users
+++++++++++++++++

Many of the packages that we use in ``orbitize`` were originally written for Linux or macOS.
For that reason, we highly recommend installing the 
`Windows Subsystem for Linux (WSL) <https://docs.microsoft.com/en-us/windows/wsl/about>`_
which is an entire Linux development environment within Windows. See `here <https://github.com/semaphoreP/codeastro/blob/main/Day0/INSTALL_WINDOWS.md#windows-subsystem-for-linux-wsl>`_ 
for a handy getting started guide.

If you don't want to use WSL, there are a few extra steps you'll need to follow
to get ``orbitize`` running:

1. There is a bug with the ``ptemcee`` installation that, as far as we know, only affects Windows users. 
To work around this, download ``ptemcee`` from `its pypi page <https://pypi.org/project/ptemcee/>`_. 
Navigate to the root ``ptemcee`` folder, remove the ``README.md`` file, then install:

.. code-block:: bash

	$ cd ptemcee
	$ rm README.md
	$ pip install . --upgrade

2. Some users have reported issues with installing ``curses``. If this happens to you, you can install 
``windows-curses`` which should work as a replacement.

.. code-block:: bash

	$ pip install windows-curses

3. Finally, ``rebound`` is not compatible with windows, so you'll need to git clone
orbitize, remove rebound from orbitize/requirements.txt, then install from
the command line. 

.. code-block:: bash

    $ git clone https://github.com/sblunt/orbitize.git
    $ cd orbitize

Open up orbitize/requirements.txt, remove ``rebound``, and save.

.. code-block:: bash

    $ pip install . --upgrade


For Developers
++++++++++++++

``orbitize`` is actively being developed. The following method for 
installing ``orbitize`` will allow you to use it and make changes to it. 
After cloning the Git repository, run the following command in the top level 
of the repo:

.. code-block:: bash
	
	$ pip install -r requirements.txt -e .

Issues?
+++++++

If you run into any issues installing ``orbitize``, please create an issue on GitHub.

If you are specifically having difficulties using ``cython`` to install ``orbitize``, we
suggest first trying to install all of the ``orbitize`` dependencies (listed in 
``requirements.txt``), then disabling compilation of the C-based Kepler module with 
the following alternative installation command:

.. code-block:: bash
	
	$ pip install orbitize --install-option="--disable-cython"



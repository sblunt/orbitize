This document should grow and change with our code base. Please revise & add anything you think is relevant!

# Things You Should Do Before Contributing:

- Learn about [object-oriented programming](http://www.voidspace.org.uk/python/articles/OOP.shtml). 
- Learn about [Markov Chain Monte Carlo algorithms](https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/). Ask Jason when you have questions.
- Learn about [Orbits for the Impatient (OFTI)](http://adsabs.harvard.edu/abs/2017AJ....153..229B). Ask Sarah when you have questions.
- Learn about [Git and GitHub](https://product.hubspot.com/blog/git-and-github-tutorial-for-beginners)
- Read our [community agreement](https://docs.google.com/document/d/1ZzjkoB20vVTlg2wbNpS7sRjmcSrECdh8kQ11-waZQhw/edit) and (optionally) suggest changes. 


# Best Practices for Contributing:

- Code according to [PEP 8 style standards](https://www.python.org/dev/peps/pep-0008/). When in doubt, check the guide!
- Apply object-oriented principles. Abstract out common functionality whenever possible!
- Document all your code using [doc strings](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- Put your name on all functions you write in the doc strings. Update contributors.txt periodically so it accurately summarizes your role in the project.
- Write unit tests for your code (see "Unit Testing Instructions" below).
- Don’t edit the `main` branch directly. Create a new branch and put all of your code there. When you’re happy with your changes, create a pull request, and assign at least Sarah and Jason to review it. When everyone is happy with it, either Sarah or Jason will pull in your changes.
- Each feature should get its own branch to keep things modular. (e.g., don't have a branch like 'Jason-changes' that is a bunch of things all at once). The person assigned to that feature is the lead of the branch.
- Ask for permission from the lead of a branch before contributing to that branch. Helping with a branch is nice, but do ask for permission first since they probably have a picture of what they want to do already. 
- Every time you make changes, include a descriptive commit message. The more detail, the better!

# Unit Testing Instructions:

When you add some code to `orbitize/orbitize`, you should also add some functions to `orbitize/tests` (in your development branch, NOT the `main` branch). Any test function you write must be named `test_*` (see existing tests for examples). You can either create a new file for your test function(s) or add them to an existing `orbitize/tests/test_*` file. 

`orbitize` uses a service called Travis Continuous Integration (CI) to run all properly named tests in `orbitize/tests` every time someone adds some commits or creates a pull request. Check out `orbitize`'s current Travis CI status [here](https://travis-ci.org/sblunt/orbitize). 

If you're interested in learning more about how Travis CI tests all of our code, check out their [site](https://docs.travis-ci.com/user/getting-started/)!

You should test as much of your code as possible (ideally all of it). To check how many lines of code in `orbitize/orbitize` are currently being tested by our unit tests, check out our [coveralls site](https://coveralls.io/github/sblunt/orbitize).

# Notes:

- Naming practices:
    - modules = all lowercase 
    - Classes = named with first letter uppercase 
    - multi-word function names = lowercase with underscores
    
- Releasing a new version:
    - Pull all new code to main
    - Increment version in \_\_init\_\_
    - Update docs changelog (remember to give credit to contributors)
    - GitHub release
    - Upload new code to PyPi
    

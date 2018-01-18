This document should grow and change with our code base. Please revise & add anything you think is relevant!

# Things You Should Do Before Contributing:

- Learn about [object-oriented programming](http://www.voidspace.org.uk/python/articles/OOP.shtml). 
- Learn about [Markov Chain Monte Carlo algorithms](https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/). Ask Jason when you have questions.
- Learn about [Orbits for the Impatient (OFTI)](http://adsabs.harvard.edu/abs/2017AJ....153..229B). Ask Sarah when you have questions.
- Learn about [Git and GitHub](https://product.hubspot.com/blog/git-and-github-tutorial-for-beginners)


# Best Practices for Contributing:

- Code according to [PEP 8 style standards](https://www.python.org/dev/peps/pep-0008/). When in doubt, check the guide!
- Apply object-oriented principles. Abstract out common functionality whenever possible!
- Document all your code using [doc strings](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- Put your name on all functions you write in the doc strings. Update contributors.txt periodically so it accurately summarizes your role in the project.
- Write unit tests for your code (Sarah will put more specific instructions here once Travis is set up)
- Don’t edit the `master` branch directly. Create a new branch and put all of your code there. When you’re happy with your changes, create a pull request, and assign at least Sarah and Jason to review it. When everyone is happy with it, either Sarah or Jason will pull in your changes.
- Every time you make changes, include a descriptive commit message. The more detail, the better!

# Notes:
- Naming practices:
    - modules = all lowercase 
    - Classes = named with first letter uppercase 
    - multi-word function names = lowercase with underscores

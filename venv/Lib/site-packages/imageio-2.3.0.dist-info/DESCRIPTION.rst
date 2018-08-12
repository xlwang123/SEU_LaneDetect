
.. image:: https://travis-ci.org/imageio/imageio.svg?branch=master
    :target: https://travis-ci.org/imageio/imageio'

.. image:: https://coveralls.io/repos/imageio/imageio/badge.png?branch=master
  :target: https://coveralls.io/r/imageio/imageio?branch=master


Imageio is a Python library that provides an easy interface to read and
write a wide range of image data, including animated images, volumetric
data, and scientific formats. It is cross-platform, runs on Python 2.7
and 3.4+, and is easy to install.

Main website: http://imageio.github.io


Release notes: http://imageio.readthedocs.io/en/latest/releasenotes.html

Example:

.. code-block:: python

    >>> import imageio
    >>> im = imageio.imread('imageio:astronaut.png')
    >>> im.shape  # im is a numpy array
    (512, 512, 3)
    >>> imageio.imwrite('astronaut-gray.jpg', im[:, :, 0])

See the `user API <http://imageio.readthedocs.io/en/latest/userapi.html>`_
or `examples <http://imageio.readthedocs.io/en/latest/examples.html>`_
for more information.



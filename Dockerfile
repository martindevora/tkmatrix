FROM sherlockpipe:sherlockpipe

RUN python3 -m pip install tkmatrix
CMD ["/bin/bash"]
FROM sherlockpipe/sherlockpipe:latest

RUN python3 -m pip install tkmatrix
CMD ["/bin/bash"]
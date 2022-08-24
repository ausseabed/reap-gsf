# reap-gsf

A prototype module for extracting the contents of a GSF file.

There were other Python modules for reading the GSF files. Some were a little convoluted,
not very pythonic, prone to crashing as they were sensitive to C types. Others required
a significant amount of code just to wrap it in order to make it friendlier to use.
The last issue was they all required a file on disk to read, and not something from a cloud
object store such as AWS S3.

Attempted to address all record types within the GSF spec.
Backscatter was unfortunately left out due to time constraints, as well as the fact
that the project was focussed on the ping depth data.

The code developed within is considered a prototype, and was developed specifically
for the [ARDC PL019 GMRT project](https://ecat.ga.gov.au/geonetwork/srv/api/records/c09a1237-f1af-4f1b-8509-4eae5636fd14).

Whilst it was used in a small-scale production sense for the project, care
must still be taken as it was designed specifically to meet the requirements
of the project.

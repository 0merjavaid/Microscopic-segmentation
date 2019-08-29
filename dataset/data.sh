# wget https://www.dropbox.com/sh/5v78g92un5smbsr/AAAQh23NyKQIg_KM0n9ghlkza/Example%20data?dl=1 -O example.zip
# unzip example.zip
# rm example.zip

wget https://www.dropbox.com/sh/5v78g92un5smbsr/AAD70oCfF39z61OJaqrSzhyra/imagepairs?dl=1 -O pairs.zip
unzip pairs.zip
mkdir imagepair
mv set* imagepair/
rm pairs.zip


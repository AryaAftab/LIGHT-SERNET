python train.py -dn "EMO-DB" -id 3 -at "all" -ln "focal" -v 1 -it "mfcc" -c "disk" -m false
python train.py -dn "EMO-DB" -id 3 -at "all" -ln "cross_entropy" -v 1 -it "mfcc" -c "disk" -m false

python train.py -dn "IEMOCAP" -id 7 -at "all" -ln "focal" -v 1 -it "mfcc" -c "disk" -m false
python train.py -dn "IEMOCAP" -id 7 -at "all" -ln "cross_entropy" -v 1 -it "mfcc" -c "disk" -m false

python train.py -dn "IEMOCAP" -id 7 -at "impro" -ln "focal" -v 1 -it "mfcc" -c "disk" -m false
python train.py -dn "IEMOCAP" -id 7 -at "impro" -ln "cross_entropy" -v 1 -it "mfcc" -c "disk" -m false

python train.py -dn "IEMOCAP" -id 3 -at "all" -ln "focal" -v 1 -it "mfcc" -c "disk" -m false
python train.py -dn "IEMOCAP" -id 3 -at "all" -ln "cross_entropy" -v 1 -it "mfcc" -c "disk" -m false

python train.py -dn "IEMOCAP" -id 3 -at "impro" -ln "focal" -v 1 -it "mfcc" -c "disk" -m false
python train.py -dn "IEMOCAP" -id 3 -at "impro" -ln "cross_entropy" -v 1 -it "mfcc" -c "disk" -m false
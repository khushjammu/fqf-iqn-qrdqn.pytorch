pip3 install gym
wget http://www.atarimania.com/roms/Roms.rar
sudo apt-get update && sudo apt-get install unrar
unrar x Roms.rar
unzip "ROMS.zip"
pip3 install atari_py
python3 -m atari_py.import_roms ROMS
pip3 install dm-reverb[tensorflow]
pip3 uninstall tensorboard==2.5.0 tensorflow-estimator==2.5.0
# sudo apt-get install python3-opencv

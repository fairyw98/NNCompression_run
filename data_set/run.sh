if [ ! -f "flower_photos.tgz" ]; then
  wget https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
fi

if [ ! -d "flower_data" ]; then
  mkdir flower_data
fi
tar -xvzf flower_photos.tgz -C flower_data
python split_data.py
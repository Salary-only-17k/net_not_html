FROM tensorflow-serving-cpu
ENV ubuntu18.04
RUN sudo docker pull tensorflow/serving:latest-devel\
	git clone https://github.com/tensorflow/serving
your_model_path='$/'
CMD docker run -p 8501:8501 \
  --mount type=bind,\
  source=$/root/software/serving/tensorflow_serving/servables/tensorflow/testdata/{your_model_path},\
  target=/models/half_plus_two \
  -e MODEL_NAME=half_plus_two -t tensorflow/serving &
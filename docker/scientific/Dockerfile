FROM python@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f
COPY wheels /wheels
COPY requirements.lock /opt/mlrescurix/requirements.lock
RUN python -m pip install --no-index --find-links=/wheels --require-hashes -r /opt/mlrescurix/requirements.lock && python -m pip check && rm -rf /wheels /root/.cache
COPY worker /worker
RUN python -m pip install --no-index --no-build-isolation --no-deps /worker && python -m pip check && rm -rf /worker
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

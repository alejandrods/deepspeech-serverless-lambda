try:
  import unzip_requirements
except ImportError:
  pass

import shlex
import subprocess
import base64
import io
import sys
import wave
import json
import numpy as np

from deepspeech import Model

ds = Model('./model/deepspeech-0.7.1-models.pbmm')
desired_sample_rate = ds.sampleRate()

try:
    from shhlex import quote
except ImportError:
    from pipes import quote


def convert_samplerate(audio_path, desired_sample_rate):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path), desired_sample_rate)
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))

    return desired_sample_rate, np.frombuffer(output, np.int16)


def inferHandler(event, context):
    body = json.loads(event['body'])

    content = base64.b64decode(body['content'])

    bytes = io.BytesIO(content)

    fin = wave.open(bytes, 'rb')
    fs_orig = fin.getframerate()
    if fs_orig != desired_sample_rate:
        fs_new, audio = convert_samplerate(bytes, desired_sample_rate)
    # else:
        print("same rate")
        # audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    text_converted = ds.stt(audio)

    response = {
        "statusCode": 200,
        "body": fs_orig,
        "text": text_converted
    }

    return response


def healthLiveness(event, context):
    response = {
        "statusCode": 200,
        "version": "0.0.1"
    }

    return response

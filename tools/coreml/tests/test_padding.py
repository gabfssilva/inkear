from conftest import Reference, relative_error

from supertonic_coreml.assets import Assets, Voice
from supertonic_coreml.pipeline import Bucket, Stages, synthesize


def describe_synthesize():
    def when_padded_to_a_bucket():
        def it_does_not_depend_on_the_bucket_size(
            reference: Reference, onnx: Stages, assets: Assets, voice: Voice
        ):
            doubled = Bucket(2 * reference.bucket.text_length, 2 * reference.bucket.latent_length)

            larger = synthesize(onnx, assets, reference.text, "pt", voice, bucket=doubled)

            assert larger.seconds == reference.speech.seconds
            assert relative_error(larger.wav, reference.speech.wav) < 1e-4

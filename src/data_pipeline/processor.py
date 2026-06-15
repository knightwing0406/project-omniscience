import torch
import torchaudio.transforms as T

class AcousticSignalProcessor:
    """
    GPU-Accelerated Signal Processing Module.
    Converts raw 1D streaming audio arrays into 2D logarithmic Mel Spectrogram matrices.
    """
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize mathematical transform layers directly on the target device
        self.mel_transformer = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        ).to(self.device)
        
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80).to(self.device)

    def process_stream_chunk(self, raw_audio_list):
        """Transforms raw incoming audio arrays into normalized torch tensor matrices."""
        audio_tensor = torch.tensor(raw_audio_list, dtype=torch.float32, device=self.device)
        audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension [1, Samples]
        
        with torch.no_grad():  # Turn off gradient tracking to maximize inference processing speed
            mel_spectrogram = self.mel_transformer(audio_tensor)
            log_mel_spectrogram = self.amplitude_to_db(mel_spectrogram)
            
        return log_mel_spectrogram

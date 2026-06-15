import math
import random
import time

class IndustrialAudioStreamer:
    """
    Simulates real-time telemetry streaming from factory microphones.
    Generates continuous acoustic wave packets for deep learning analysis.
    """
    def __init__(self, sample_rate: int = 16000, chunk_duration_ms: int = 500):
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.samples_per_chunk = int(self.sample_rate * (self.chunk_duration_ms / 1000))
        self.baseline_frequency = 60.0  # 60 Hz baseline motor hum
        self.time_accumulator = 0.0

    def generate_live_packet(self):
        """Generates a continuous time-series audio packet, injecting faults at a 10% rate."""
        packet_data = []
        has_anomaly = random.random() < 0.10
        
        for _ in range(self.samples_per_chunk):
            self.time_accumulator += 1.0 / self.sample_rate
            healthy_wave = math.sin(2 * math.pi * self.baseline_frequency * self.time_accumulator)
            ambient_noise = random.gauss(0, 0.15)
            combined_signal = healthy_wave + ambient_noise
            
            if has_anomaly:
                # 1200 Hz metal-on-metal friction strike
                friction_shriek = math.sin(2 * math.pi * 1200.0 * self.time_accumulator) * 0.75
                combined_signal += friction_shriek
                
            packet_data.append(round(combined_signal, 4))
            
        return packet_data, has_anomaly

if __name__ == "__main__":
    streamer = IndustrialAudioStreamer()
    print("⚡ [INGESTION] Live Audio Stream Ingestion Online.\n")
    for block in range(1, 4):
        data, flag = streamer.generate_live_packet()
        status = "🚨 ANOMALY" if flag else "✅ NORMAL"
        print(f"📦 [Packet {block}] Buffered {len(data)} samples | Status: {status}")
        time.sleep(0.1)

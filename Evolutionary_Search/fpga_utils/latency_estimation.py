import numpy as np

class LatencyTable():
    def __init__ (self, path = '../datasets/latency_lut.npy'):
        self.path = path 
        self.efficiency_dict = np.load(self.path, allow_pickle=True).item()
        
        
    #exception -- I have not included the latency of average pooling 
    def predict_efficiency(self, sample):
        input_size = sample.get("r", [224])
        input_size = input_size[0]
        assert "ks" in sample and "e" in sample and "d" in sample
        assert len(sample["ks"]) == len(sample["e"]) and len(sample["ks"]) == 20
        assert len(sample["d"]) == 5
        
        total_latency = 0 
        
        for i in range(20):
            stage = i // 4
            depth_max = sample["d"][stage]
            depth = i % 4 + 1
            if depth > depth_max:
                continue
            ks, e = sample["ks"][i], sample["e"][i]
            
            
            total_latency+= self.efficiency_dict["mobile_inverted_blocks"][i+1][(ks, e)]
            


        for key in self.efficiency_dict["other_blocks"]:
            total_latency+=self.efficiency_dict["other_blocks"][key]
            
        
        return total_latency
    
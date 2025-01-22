import matplotlib.pyplot as plt

def show_results(pkl_path):
    # ... existing code ...

    for i in range(10):
        # ... existing code ...

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(true_h, 'g', label='true')
        ax1.plot(pred_h, 'lightblue', label='pred')
        ax1.set_ylabel('Hue')
        ax1.set_title(file_names[i])
        ax1.legend()
        ax1.set_ylim(0, 179)  # 设置 Hue 的 y 轴范围
        
        ax2.plot(true_v, 'g', label='true')
        ax2.plot(pred_v, 'lightblue', label='pred')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.set_ylim(0, 255)  # 设置 Value 的 y 轴范围
        
        plt.tight_layout()
        # ... existing code ... 
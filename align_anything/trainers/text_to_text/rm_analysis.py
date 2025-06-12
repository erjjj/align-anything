import json
import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from align_anything.trainers.text_to_text.rm import RMTrainer

class RMAnalysisTrainer(RMTrainer):
    """Trainer for analyzing reward model predictions."""

    def __init__(self, cfgs, ds_cfgs) -> None:
        super().__init__(cfgs, ds_cfgs)
        self.scores_dict = {
            'chosen': [],
            'rejected': [],
            'prompts': [],
        }

    def eval_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Run evaluation on a batch of data."""
        info = super().eval_step(batch)
        
        # Get raw scores 
        with torch.no_grad():
            chosen_scores = self.get_rewards(
                prompt_ids=batch['prompt_ids'],
                prompt_attention_mask=batch['prompt_attention_mask'],
                response_ids=batch['chosen_response_ids'],
                response_attention_mask=batch['chosen_response_attention_mask'],
            )
            rejected_scores = self.get_rewards(
                prompt_ids=batch['prompt_ids'], 
                prompt_attention_mask=batch['prompt_attention_mask'],
                response_ids=batch['rejected_response_ids'],
                response_attention_mask=batch['rejected_response_attention_mask'],
            )

        # Convert tensor to list and save scores
        chosen_scores = chosen_scores.cpu().tolist()
        rejected_scores = rejected_scores.cpu().tolist()
        
        # Get prompts
        prompts = self.tokenizer.batch_decode(
            batch['prompt_ids'],
            skip_special_tokens=True,
        )
        
        self.scores_dict['chosen'].extend(chosen_scores)
        self.scores_dict['rejected'].extend(rejected_scores)
        self.scores_dict['prompts'].extend(prompts)
        
        return info

    def plot_score_distribution(self) -> None:
        """Plot the distribution of scores."""
        chosen_scores = np.array(self.scores_dict['chosen'])
        rejected_scores = np.array(self.scores_dict['rejected'])
        
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=chosen_scores, label='Chosen', color='green')
        sns.kdeplot(data=rejected_scores, label='Rejected', color='red')
        plt.xlabel('Reward Score')
        plt.ylabel('Density')
        plt.title('Distribution of Reward Scores')
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(self.cfgs.logger_cfgs.output_dir, 'score_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        
        # Calculate and print statistics
        stats = {
            'chosen_mean': float(np.mean(chosen_scores)),
            'chosen_std': float(np.std(chosen_scores)),
            'rejected_mean': float(np.mean(rejected_scores)),
            'rejected_std': float(np.std(rejected_scores)),
            'score_gap': float(np.mean(chosen_scores - rejected_scores)),
        }
        
        # Save scores and stats
        output_file = os.path.join(self.cfgs.logger_cfgs.output_dir, 'rm_analysis.json')
        with open(output_file, 'w') as f:
            json.dump({
                'scores': self.scores_dict,
                'stats': stats,
            }, f, indent=2, ensure_ascii=False)
        
        self.logger.print('\nAnalysis Results:')
        self.logger.print(f'Chosen responses - Mean: {stats["chosen_mean"]:.3f}, Std: {stats["chosen_std"]:.3f}')
        self.logger.print(f'Rejected responses - Mean: {stats["rejected_mean"]:.3f}, Std: {stats["rejected_std"]:.3f}')
        self.logger.print(f'Average score gap: {stats["score_gap"]:.3f}')

    def eval(self) -> Dict[str, float]:
        """Run evaluation and generate visualizations."""
        results = super().eval()
        self.plot_score_distribution()
        return results

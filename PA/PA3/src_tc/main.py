#!/usr/bin/env python3
"""
Main entry point for PyTorch-based RL training
"""

import argparse
import logging
import torch

from getup import train_ppo_getup
from walk import train_ppo_walk

def main():
    parser = argparse.ArgumentParser(description='PyTorch RL Training')
    parser.add_argument('--task', choices=['getup', 'walk'], required=True,
                       help='Task to train on')
    parser.add_argument('--fast', action='store_true',
                       help='Use fast training mode (fewer episodes)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='auto',
                       help='Device to use for training')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Run training
    if args.task == 'getup':
        logger.info("Starting getup task training...")
        agent, rewards, error = train_ppo_getup(fast_mode=args.fast)
        logger.info(f"Getup training completed. Height error: {error:.3f}")
        
    elif args.task == 'walk':
        logger.info("Starting walk task training...")
        agent, rewards, error = train_ppo_walk()
        logger.info(f"Walk training completed. Velocity error: {error:.3f}")
    
    logger.info("Training completed successfully!")

if __name__ == '__main__':
    main()
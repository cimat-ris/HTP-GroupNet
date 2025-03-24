import argparse
import os
import sys
sys.path.append(os.getcwd())
import torch
from model.GroupNet_nba import GroupNet


def main():
    # Configuración
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='saved_models/nba/pretrain.p', help='Ruta al modelo entrenado')
    args = parser.parse_args()

    # Cargar modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando dispositivo: {device}')
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model = GroupNet(checkpoint['model_cfg'], device=device)
    model.set_device(device)
    model.load_state_dict(checkpoint['model_dict'])
    model.eval()

    # Crear dato manual (batch_size=1, 11 jugadores, 5 timesteps pasados, 2 coordenadas)
    input_data = {
        'past_traj': torch.randn(32, 11, 5, 2).to(device),  
        'seq': ['manual_input']
    }

    with torch.no_grad():
        prediction = model.inference(input_data)
    
    # Resultados
    print("\nEntrada (past_traj):")
    print(input_data['past_traj'].shape)
    print("\nPredicción (future_traj):")
    print(prediction.shape)

if __name__ == '__main__':
    main()
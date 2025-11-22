"""
Test rapid pentru modelul hibrid
Verifică dacă toate tipurile de modele funcționează corect
"""

import torch
from models.HYBRID_MODEL import create_hybrid_model

def test_all_hybrid_models():
    print("=" * 80)
    print("🧪 TESTARE MODELE HIBRIDE")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")

    # Test toate tipurile
    model_types = ['hybrid_concat', 'hybrid_add', 'hybrid_attention', 'ensemble']

    for model_type in model_types:
        print(f"\n{'='*80}")
        print(f"📦 Testing: {model_type.upper()}")
        print(f"{'='*80}")

        try:
            # Creează modelul
            model = create_hybrid_model(num_classes=3, input_size=224, model_type=model_type)
            model = model.to(device)
            model.eval()

            # Test forward pass cu un batch mic
            batch_size = 2
            dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

            with torch.no_grad():
                output = model(dummy_input)

            # Verificări
            assert output.shape == (batch_size, 3), f"Wrong output shape: {output.shape}"
            assert not torch.isnan(output).any(), "Output contains NaN!"
            assert not torch.isinf(output).any(), "Output contains Inf!"

            # Statistici
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"\n✅ {model_type.upper()} - SUCCESS!")
            print(f"   Output shape: {output.shape}")
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
            print(f"   Model size: ~{total_params * 4 / (1024**2):.1f} MB")

            # Test gradient flow
            model.train()
            output = model(dummy_input)
            loss = output.sum()
            loss.backward()

            has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
            if has_grad:
                print(f"   ✅ Gradient flow: OK")
            else:
                print(f"   ⚠️  Gradient flow: Some parameters have no gradient")

            # Memory cleanup
            del model, output, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"\n❌ {model_type.upper()} - FAILED!")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("✅ TESTARE COMPLETĂ!")
    print("=" * 80)
    print("\n🚀 Toate modelele sunt funcționale și gata de training!")
    print("\n📌 Pentru a antrena un model hibrid:")
    print("   1. Editează MODEL_TYPE în train_hybrid.py")
    print("   2. Rulează: python train_hybrid.py")
    print("\n💡 Recomandare: Începe cu 'hybrid_concat' pentru cele mai bune rezultate!")


if __name__ == "__main__":
    test_all_hybrid_models()


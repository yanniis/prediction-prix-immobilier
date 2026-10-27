"""Réentraîne les 4 modèles du projet, dans l'ordre (le type de logement dépend du clustering)."""
import sys

import train_clustering
import train_prix
import train_recommandation
import train_type_logement

if __name__ == "__main__":
    print("\n=== 1. Estimation du prix ===");      train_prix.main(grid="--grid" in sys.argv)
    print("\n=== 2. Profils (clustering) ===");    train_clustering.main()
    print("\n=== 3. Type de logement ===");        train_type_logement.main()
    print("\n=== 4. Recommandation ===");         train_recommandation.main()
    print("\nModèles enregistrés dans le dossier models/.")

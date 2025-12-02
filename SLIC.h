#ifndef MIDTERMASSIGNMENT_SLIC_H
#define MIDTERMASSIGNMENT_SLIC_H
#endif

#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>

//
//
//
//
//Usiamo una struttura dedicata per i centri, che include gli accumulatori
// per calcolare in modo efficiente la nuova media del cluster.
struct ClusterCenter {
    float l, a, b; // Valori LAB del centro
    int x, y;     // Posizione spaziale del centro

    // Accumulatori per il ricalcolo del baricentro
    long double sum_l, sum_a, sum_b;
    long long sum_x, sum_y;
    int pixel_count; // Numero di pixel assegnati al cluster

    ClusterCenter() :
        l(0), a(0), b(0), x(0), y(0),
        sum_l(0), sum_a(0), sum_b(0), sum_x(0), sum_y(0), pixel_count(0) {}
};

class SLIC {
private:
    cv::Mat image; // Immagine LAB a 3 canali (float)
    int width, height;
    int N; // Numero totale di pixel (width * height)



    // Parametri SLIC
    int num_superpixels; // K
    float compactness; // m (variabile per il controllo della compattezza)
    int S; // Intervallo della griglia. dato da sqrt(N/K)

    const int MAX_ITERATIONS = 10; // Numero di max.iterazioni
    const int N_NEIGHBORHOOD = 3; // Dimensione del vicinato (3x3) per la perturbazione iniziale




    // Dati per l'algoritmo
    std::vector<ClusterCenter> centers; // Centri di cluster "Ck"
    std::vector<int> labels; // Etichetta (indice del cluster) per ogni pixel (piatta: width * height)
    std::vector<float> distances; // Distanza minima D_s per ogni pixel (piatta: width * height)

    // Funzioni helper
    void convert_to_lab();
    void perturb_centers();
    float calculate_gradient(int x, int y);
    float calculate_distance(int center_k, int p_x, int p_y);
    void compute_new_centers();

public:
    // Costruttore: K è num_superpixels, m è compactness
    SLIC(int num_superpixels, float compactness);

    // Metodi pubblici
    bool load_image(const std::string& path);
    void initialize_clusters();
    void assign_clusters();
    void update_clusters();
    void enforce_connectivity();
    void save_result(const std::string& path, const std::string& mode = "boundaries");
    void process();
};


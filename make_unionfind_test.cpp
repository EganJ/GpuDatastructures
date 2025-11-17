/**
 * File for generating a random set of union-find operations and saving them to a file
 * that can be used for repeatable testing.
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <math.h>
#include <algorithm>
/** Classes are in the range [1, N], and store 0 when they are root otherwise
 * the class ID of their parent. The first position in the array is left blank
 * to avoid confusion.
 *
 * Creates a directory with the following files:
 *  - expected.txt: contains the expected output of the union-find, except
 *    that the first position (normally unused) is used to store N.
 *  - mergers.txt: contains the value of num_mergers followed by two arrays of length
 *    num_mergers, the first containing the 'a' values and the second containing
 *    the 'b' values.
 */

struct CPUUnionFind
{
    std::vector<int> classes;
    int n_classes;
    CPUUnionFind(int n) : n_classes(n)
    {
        classes.resize(n + 1);
        for (int i = 0; i <= n; ++i)
        {
            classes[i] = 0;
        }
    }
    int get_class(int i)
    {
        if (classes[i] == 0)
            return i;
        int root = get_class(classes[i]);
        classes[i] = root; // path compression
        return root;
    }
    void merge(int a, int b)
    {
        a = get_class(a);
        b = get_class(b);
        if (a == b)
            return;
        if (a > b)
            std::swap(a, b);
        classes[b] = a; // merge b into a
        n_classes--;
    }

    void finalize_flatten_all()
    {
        for (int i = 1; i < classes.size(); ++i)
        {
            get_class(i);
        }
    }
};

void generate_test(int num_elements, std::string output_path)
{
    // Simply generating random merges will almost certainly result in a single giant class
    // with maybe a couple tiny stragglers. To prevent this and have a more interesting test,
    // sort into partitions before hand and only allow merges within partitions.
    std::vector<std::vector<int>> partitions;
    int n_roots = std::sqrt(num_elements);
    for (int i = 0; i < n_roots; ++i)
    {
        partitions.push_back(std::vector<int>());
    }
    for (int i = 1; i <= num_elements; ++i)
    {
        // Randomly assign to a partition
        int p = rand() % n_roots;
        partitions[p].push_back(i);
    }

    std::vector<std::pair<int, int>> merges;
    // Create 1 merge per element, relying on randomness to provide some elements with multiple merges.
    for (const auto &p : partitions)
    {
        for (int i = 0; i < p.size(); ++i)
        {
            int a = p[rand() % p.size()];
            int b = p[rand() % p.size()];
            merges.push_back(std::make_pair(a, b));
        }
        // It's ok if we don't end up with a connected partition, since
        // this was created just to ensure we don't end up with a single giant class.
    }

    // Randomize the order of merges
    std::shuffle(merges.begin(), merges.end(), std::mt19937{std::random_device{}()});

    std::vector<int> mergers_a;
    std::vector<int> mergers_b;
    CPUUnionFind uf(num_elements);
    // Perform merges on uf to get expected output
    for (const auto &m : merges)
    {
        uf.merge(m.first, m.second);
        mergers_a.push_back(m.first);
        mergers_b.push_back(m.second);
    }

    uf.finalize_flatten_all();

    std::ofstream expected_file(output_path + "/expected.data");
    expected_file.write((char *)&num_elements, sizeof(num_elements));
    int *expected_classes = uf.classes.data() + 1; // skip first element
    expected_file.write((char *)expected_classes, sizeof(int) * (num_elements));
    expected_file.close();

    std::ofstream mergers_file(output_path + "/mergers.data");
    int num_mergers = mergers_a.size();
    mergers_file.write((char *)&num_mergers, sizeof(num_mergers));
    mergers_file.write((char *)mergers_a.data(), sizeof(int) * num_mergers);
    mergers_file.write((char *)mergers_b.data(), sizeof(int) * num_mergers);
    mergers_file.close();
}

int main(int argc, char *argv[])
{
    // Args: num_elements, max_mergers, output_path
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " num_elements output_path" << std::endl;
        return 1;
    }
    int num_elements = atoi(argv[1]);
    std::string output_path = argv[2];
    generate_test(num_elements, output_path);
    return 0;
}
#include <bits/stdc++.h>

class CNN
{
private:
    std::vector<std::string> train;
    std::vector<std::string> test;

    std::vector<int> train_y;
    std::vector<int> train_x;

    std::vector<int> test_y;
    std::vector<int> test_x;

    std::string train_name = "fashion-mnist_train.csv";
    std::string test_name = "fashion-mnist_test.csv";

public:
    CNN()
    {
        // load data
        this->read_data(this->train_name, this->train);
        this->read_data(this->test_name, this->test);

        // 
        
        // print data
        // this->display(this->train_name, this->train);
        // this->display(this->test_name, this->test);
    }

    void read_data(const std::string& file_name, std::vector<std::string>& data) 
    {
        std::ifstream file(file_name);
        if (!file.is_open())
        {
            std::cerr << "Could not open the file: " << file_name << std::endl;
            return;
        }

        std::string line;
        while (std::getline(file, line))
        {
            data.push_back(line);
        }

        file.close();
    }

    void X(std::vector<int> &X)
    {

    }

    void display(const std::string& file_name, const std::vector<std::string>& data)
    {
        std::cout << "Displaying contents of " << file_name << ":\n";

        for (const std::string& row : data)
        {
            std::cout << row << "\n";
        }
    }
};

int main()
{
    CNN object;
}

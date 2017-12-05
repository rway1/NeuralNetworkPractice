using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkPractice
{
    class Program
    {
        static void Main(string[] args)
        {
            List<List<double>> x = new List<List<double>>
            {
                new List<double>
                {
                    0.0,
                    0.0
                },
                new List<double>
                {
                    1.0,
                    0.0
                },
                new List<double>
                {
                    0.0,
                    1.0
                },
                new List<double>
                {
                    1.0,
                    1.0
                },
            };
            List<List<double>> y = new List<List<double>>
            {
                new List<double>
                {
                    0.0,
                    0.0
                },
                new List<double>
                {
                    1.0,
                    1.0
                },
                new List<double>
                {
                    1.0,
                    1.0
                },
                new List<double>
                {
                    1.0,
                    1.0
                }
            };
            NeuralNetwork neuralNetwork = new NeuralNetwork();
            neuralNetwork.Train(x, y);
            List<double> list = neuralNetwork.Test(0, 0);
            Console.WriteLine("Results for 0,0: {0}", HelperFunctions.ListToString(list));
            list = neuralNetwork.Test(1, 0);
            Console.WriteLine("Results for 1,0: {0}", HelperFunctions.ListToString(list));
            list = neuralNetwork.Test(0, 1);
            Console.WriteLine("Results for 0,1: {0}", HelperFunctions.ListToString(list));
            list = neuralNetwork.Test(1, 1);
            Console.WriteLine("Results for 1,1: {0}", HelperFunctions.ListToString(list));
            Console.ReadKey();
        }
    }
}

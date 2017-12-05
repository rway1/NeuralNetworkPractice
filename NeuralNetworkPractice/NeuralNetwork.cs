using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkPractice
{
    class NeuralNetwork
    {
        #region ClassFields
        int currentInput;
        List<double> a; //node results
        List<List<double>> x; //input data
        List<double> j; //Node error
        List<List<double>> w; //weights
        List<List<double>> y; //expected output
        int layerDepth;
        int layerCount;
        double alpha;
        #endregion

        public NeuralNetwork()
        {
            InitializeFields();
        }

        #region NodeMathFunctions
        /// <summary>
        /// Activation function for nodes
        /// </summary>
        /// <param name="In"></param>
        /// <returns name="result"></returns>
        double G(double In)
        {
            return 1 / (1 + Math.Pow(Math.E, -In));
        }

        /// <summary>
        /// Dotproduct function
        /// </summary>
        /// <param name="a"></param>
        /// <param name="w"></param>
        /// <returns></returns>
        double DotProduct(List<double> a, List<double> w)
        {
#if DEBUG
            Console.WriteLine("\nDotProduct\n"); 
#endif
            if (a.Count != w.Count)
            {
                throw new Exception("Dot product attempted on lists of unequal lenght");
            }
            double sum = 0;
            for (int count = 0; count < a.Count; count++)
            {
                sum += a[count] * w[count];
            }
#if DEBUG
            Console.WriteLine("A: {0}", HelperFunctions.ListToString(a));
            Console.WriteLine("W: {0}", HelperFunctions.ListToString(w));
            Console.WriteLine("Result: {0}", sum);
#endif
            return sum;
        } 
        #endregion

        void InitializeFields()
        {
#if DEBUG
            Console.WriteLine("\n\nInitalizeFields\n"); 
#endif
            currentInput = 0;
            a = Enumerable.Repeat(0.0, 7).ToList();
            a[0] = 1;
            x = new List<List<double>>
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
            y = new List<List<double>>
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
            j = Enumerable.Repeat(0.0, 6).ToList();
            w = new List<List<double>>();
            List<double> temp = new List<double>();
            temp.Add(0);
            for (int i = 0; i < 6; i++)
            {
                temp.Add(0.01);
            }
            w.Add(temp);
            for (int l = 1; l < 7; l++)
            {
                List<double> list = new List<double>();
                for (int k = 0; k < 7; k++)
                {
                    list.Add(0);
                }
                w.Add(list);
            }
            w[1][3] = -0.04;
            w[1][4] = -0.03;
            w[2][3] = 0.02;
            w[2][4] = 0.05;
            w[3][5] = -0.02;
            w[3][6] = -0.01;
            w[4][5] = 0.03;
            w[4][6] = 0.04;
#if DEBUG
            Console.WriteLine("W: {0}", HelperFunctions.ListToString(w));
#endif
            layerDepth = x[currentInput].Count;
            layerCount = a.Count / layerDepth;
            alpha = 0.01;
        }

        #region Training
        public void Train()
        {
            double max = 0.0;
            do
            {
                for (int i = 0; i < x.Count; i++)
                {
                    currentInput = i;
                    FeedForward();
                    BackPropagation();
                }
                max = 0.0;
                for (int i = 0; i < layerDepth; i++)
                {
                    if (j[j.Count-i-1] > max)
                    {
                        max = j[j.Count - i - 1];
                    }
                }
            } while (max > 0.001);
        }

        void FeedForward()
        {
            InitializeInputLayer();
            CalculateNodeValues();
        }

        void InitializeInputLayer()
        {
            for (int input = 0; input < x[currentInput].Count; input++)
            {
                a[input + 1] = x[currentInput][input];
            }
        }

        void CalculateNodeValues()
        {
#if DEBUG
            Console.WriteLine("\n\nCalculateNodeValues\n");
#endif
            for (int currentNode = layerDepth + 1; currentNode < a.Count; currentNode++)
            {
                List<double> list = HelperFunctions.ListColumn(w, currentNode);
                a[currentNode] = G(DotProduct(a, list));
            }
#if DEBUG
            Console.WriteLine("A iteration result: {0}", HelperFunctions.ListToString(a));
#endif
        }

        void BackPropagation()
        {
            CalculateOutputLayerError();
            CalculateHiddenLayerError();
            WeightCorrection();
        }

        void CalculateOutputLayerError()
        {
#if DEBUG
            Console.WriteLine("\n\nCalculateOutputLayerError\n");
#endif
            for (int currentNode = j.Count - 1; currentNode > j.Count - layerDepth - 1; currentNode--)
            {
                j[currentNode] = a[currentNode + 1] * (1 - a[currentNode + 1]) * (y[currentInput][currentNode % layerDepth] - a[currentNode + 1]);
            }
#if DEBUG
            Console.WriteLine("J: {0}", HelperFunctions.ListToString(j));
#endif
        }

        void CalculateHiddenLayerError()
        {
#if DEBUG
            Console.WriteLine("\n\nCalculateHiddenLayerError\n");
#endif
            for (int currentNode = j.Count - layerDepth - 1; currentNode >= 0; currentNode--)
            {
                j[currentNode] = a[currentNode + 1] * (1 - a[currentNode + 1]) * DotProduct(j, w[currentNode + 1].GetRange(1, w[currentNode + 1].Count - 1));
            }
#if DEBUG
            Console.WriteLine("J: {0}", HelperFunctions.ListToString(j));
#endif
        }

        void WeightCorrection()
        {
#if DEBUG
            Console.WriteLine("\n\nWeightCorrection\n");
#endif
            for (int h = 0; h < w.Count; h++)
            {
                for (int i = 0; i < w[h].Count - 1; i++)
                {
                    if (w[h][i + 1] != 0)
                    {
#if DEBUG
                        Console.Write("{0} + {1} * {2} * {3}", w[h][i + 1], alpha, a[h], j[i]);
#endif
                        w[h][i + 1] = w[h][i + 1] + alpha * a[h] * j[i];
#if DEBUG
                        Console.WriteLine(" = {0}", w[h][i + 1]);
#endif 
                    }
                }
            }
#if DEBUG
            Console.WriteLine("\nW: {0}", HelperFunctions.ListToString(w));
#endif
        }
        #endregion

        #region Testing
        public List<double> Test(double x, double y)
        {
            List<double> list = new List<double>
            {
                x,
                y
            };
            FeedForward(list);
            return a.GetRange(a.Count - layerDepth, layerDepth);
        }

        private void FeedForward(List<double> list)
        {
            for (int i = 0; i < list.Count; i++)
            {
                x[0][i] = list[i];
            }
            currentInput = 0;
            FeedForward();
        }
        #endregion
    }
}

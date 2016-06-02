#ifndef SamuQl_H
#define SamuQl_H

/**
 * @brief NAHSHON QL from https://github.com/nbatfai/nahshon
 *
 * @file ql.hpp
 * @author  Norbert Bátfai <nbatfai@gmail.com>
 * @version 0.0.1
 *
 * @section LICENSE
 *
 * Copyright (C) 2015, 2016 Norbert Bátfai, batfai.norbert@inf.unideb.hu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @section DESCRIPTION
 *
 * JACOB, https://github.com/nbatfai/jacob
 *
 * "The son of Isaac is Jacob." The project called Jacob is an experiment
 * to replace Isaac's (GUI based) visual imagination with a character console.
 *
 * ISAAC, https://github.com/nbatfai/isaac
 *
 * "The son of Samu is Isaac." The project called Isaac is a case study
 * of using deep Q learning with neural networks for predicting the next
 * sentence of a conversation.
 *
 * SAMU, https://github.com/nbatfai/samu
 *
 * The main purpose of this project is to allow the evaluation and
 * verification of the results of the paper entitled "A disembodied
 * developmental robotic agent called Samu Bátfai". It is our hope
 * that Samu will be the ancestor of developmental robotics chatter
 * bots that will be able to chat in natural language like humans do.
 *
 */

#include <iostream>
#include <cstdarg>
#include <map>
#include <iterator>
#include <cmath>
#include <random>
#include <limits>
#include <fstream>
#include <cstring>
#include <algorithm>

#include "orchmach1.hpp"

class Perceptron
{
public:
    Perceptron ( int nof, ... ) {
        n_layers = nof;

        units = new double*[n_layers];
        n_units = new int[n_layers];

        va_list vap;

        va_start ( vap, nof );

        for ( int i {0}; i < n_layers; ++i ) {
            n_units[i] = va_arg ( vap, int );

            if ( i ) {
                units[i] = new double [n_units[i]];
            }
        }

        va_end ( vap );

        weights = new double**[n_layers-1];

#ifndef RND_DEBUG
        std::random_device init;
        std::default_random_engine gen {init() };
#else
        std::default_random_engine gen;
#endif

        std::uniform_real_distribution<double> dist ( -1.0, 1.0 );

        for ( int i {1}; i < n_layers; ++i ) {
            weights[i-1] = new double *[n_units[i]];

            for ( int j {0}; j < n_units[i]; ++j ) {
                weights[i-1][j] = new double [n_units[i-1]];

                for ( int k {0}; k < n_units[i-1]; ++k ) {
                    weights[i-1][j][k] = dist ( gen );
                }
            }
        }
    }

    Perceptron ( std::fstream & file ) {
        file >> n_layers;

        units = new double*[n_layers];
        n_units = new int[n_layers];

        for ( int i {0}; i < n_layers; ++i ) {
            file >> n_units[i];

            if ( i ) {
                units[i] = new double [n_units[i]];
            }
        }

        weights = new double**[n_layers-1];

        for ( int i {1}; i < n_layers; ++i ) {
            weights[i-1] = new double *[n_units[i]];

            for ( int j {0}; j < n_units[i]; ++j ) {
                weights[i-1][j] = new double [n_units[i-1]];

                for ( int k {0}; k < n_units[i-1]; ++k ) {
                    file >> weights[i-1][j][k];
                }
            }
        }
    }


    double sigmoid ( double x ) {
        return 1.0/ ( 1.0 + exp ( -x ) );
    }


    double operator() ( double image [] ) {

        units[0] = image;

        for ( int i {1}; i < n_layers; ++i ) {

#ifdef CUDA_PRCPS

            //cuda_layer ( i, n_units, units, weights );

#else

            #pragma omp parallel for
            for ( int j = 0; j < n_units[i]; ++j ) {
                units[i][j] = 0.0;

                for ( int k = 0; k < n_units[i-1]; ++k ) {
                    units[i][j] += weights[i-1][j][k] * units[i-1][k];
                }

                units[i][j] = sigmoid ( units[i][j] );

            }

#endif

        }

        return sigmoid ( units[n_layers - 1][0] );

    }

    void learning ( double image [], double q, double prev_q ) {
        double y[1] {q};

        learning ( image, y );
    }

    void learning ( double image [], double y[] ) {
        //( *this ) ( image );

        units[0] = image;

        double ** backs = new double*[n_layers-1];

        for ( int i {0}; i < n_layers-1; ++i ) {
            backs[i] = new double [n_units[i+1]];
        }

        int i {n_layers-1};

        for ( int j {0}; j < n_units[i]; ++j ) {
            backs[i-1][j] = sigmoid ( units[i][j] ) * ( 1.0-sigmoid ( units[i][j] ) ) * ( y[j] - units[i][j] );

            for ( int k {0}; k < n_units[i-1]; ++k ) {
                weights[i-1][j][k] += ( 0.2* backs[i-1][j] *units[i-1][k] );
            }

        }

        for ( int i {n_layers-2}; i >0 ; --i ) {

            #pragma omp parallel for
            for ( int j =0; j < n_units[i]; ++j ) {

                double sum = 0.0;

                for ( int l = 0; l < n_units[i+1]; ++l ) {
                    sum += 0.19*weights[i][l][j]*backs[i][l];
                }

                backs[i-1][j] = sigmoid ( units[i][j] ) * ( 1.0-sigmoid ( units[i][j] ) ) * sum;

                for ( int k = 0; k < n_units[i-1]; ++k ) {
                    weights[i-1][j][k] += ( 0.19* backs[i-1][j] *units[i-1][k] );
                }
            }
        }

        for ( int i {0}; i < n_layers-1; ++i ) {
            delete [] backs[i];
        }

        delete [] backs;

    }

    ~Perceptron() {
        for ( int i {1}; i < n_layers; ++i ) {
            for ( int j {0}; j < n_units[i]; ++j ) {
                delete [] weights[i-1][j];
            }

            delete [] weights[i-1];
        }

        delete [] weights;

        for ( int i {0}; i < n_layers; ++i ) {
            if ( i ) {
                delete [] units[i];
            }
        }

        delete [] units;
        delete [] n_units;

    }

    void save ( std::fstream & out ) {
        out << " "
            << n_layers;

        for ( int i {0}; i < n_layers; ++i ) {
            out << " " << n_units[i];
        }

        for ( int i {1}; i < n_layers; ++i ) {
            for ( int j {0}; j < n_units[i]; ++j ) {
                for ( int k {0}; k < n_units[i-1]; ++k ) {
                    out << " "
                        << weights[i-1][j][k];

                }
            }
        }

    }

private:
    Perceptron ( const Perceptron & );
    Perceptron & operator= ( const Perceptron & );

    int n_layers;
    int* n_units;
    double **units;
    double ***weights;

};

#ifdef FEELINGS
typedef std::string Feeling;
#endif

//typedef std::tuple<int, int, int, int, int, int, int, int, int> Config5;
typedef std::vector<int> ConfigN;
typedef ConfigN SPOTriplet;

//typedef int SPOTriplet;
//typedef std::pair<std::string, SPOTriplet> ReinforcedAction;
typedef std::pair<int, int> ReinforcedAction;

class QL
{
public:
    /*
      QL () :tree ( &root ) {

    #ifdef FEELINGS
          std::random_device init;
          std::default_random_engine gen {init() };
          //std::uniform_int_distribution<int> dist ( 0, 1 );
          //std::uniform_int_distribution<int> dist ( 0, 6 );
          std::uniform_int_distribution<int> dist ( 0, 9 );
          //std::uniform_int_distribution<int> dist (48, 126 );
          for ( int i {0}; i<nrows; ++i ) {
              std::stringstream ss;
              for ( int j {0}; j<10; ++j ) {
                  //char c = dist ( gen );
                  //ss << c;
                  ss << dist ( gen );
              }
              prcps_f[ss.str()] = new Perceptron ( 3, 10*80, 16,  1 ); //exp.a1 // 302
          }
    #endif
      }
    */

    /*
      QL ( SPOTriplet triplet )
      {}
      */
    QL ( ) {
        for ( int prg {-1}; prg<5; ++prg )
            for ( int i {0}; i<5*2*3; ++i ) {
                table_[i][prg] = 0.0;
            }

        for ( int prg {99}; prg<105; ++prg )
            for ( int i {0}; i<5*2*3; ++i ) {
                table_[i][prg] = 0.0;
            }

    }

    ~QL() {
#ifndef Q_LOOKUP_TABLE
        for ( std::map<SPOTriplet, Perceptron*>::iterator it=prcps.begin(); it!=prcps.end(); ++it ) {
            delete it->second;
        }
#endif
#ifdef FEELINGS
        for ( std::map<Feeling, Perceptron*>::iterator it=prcps_f.begin(); it!=prcps_f.end(); ++it ) {
            delete it->second;
        }
#endif

    }

    double f ( double u, int n ) {
        if ( n < N_e ) {
            return max_reward;
        } else {
            return u;
        }
    }

#ifdef QNN_DEBUG_BREL
    int get_action_count() const {
        return frqs.size();
    }

    int get_action_relevance() const {
        /*
        double whole;
        modf (relevance*100.0 , &whole);
        return whole;
        */
        return relevance*100.0;
    }
#endif

#ifndef Q_LOOKUP_TABLE

    double max_ap_Q_sp_ap ( double image[] ) {
        double q_spap;
        double min_q_spap = -std::numeric_limits<double>::max();

        for ( std::map<SPOTriplet, Perceptron*>::iterator it=prcps.begin(); it!=prcps.end(); ++it ) {

            q_spap = ( * ( it->second ) ) ( image );
            if ( q_spap > min_q_spap ) {
                min_q_spap = q_spap;
            }
        }

        return min_q_spap;
    }

#ifdef LZW_TREE
    double max_ap_Q_sp_ap_lzw ( double image[] ) {
        double q_spap;
        double min_q_spap = -std::numeric_limits<double>::max();

        std::map<SPOTriplet, TripletNode*> children = tree->getChildren();

        int rN = children.size();

        //std::uniform_int_distribution<int> zdist ( 0, rN+1+rN/10 );
        //std::uniform_int_distribution<int> zdist ( 0, 100 );
        //if ( zdist ( zgen ) < rN )
        //if ( rN && zdist ( zgen ) < 95)
        if ( rN )
            for ( std::map<SPOTriplet, TripletNode*>::iterator it=children.begin(); it!=children.end(); ++it ) {

                q_spap = ( * ( prcps[it->first] ) ) ( image );
                if ( q_spap > min_q_spap ) {
                    min_q_spap = q_spap;
                }
            }

        else
            for ( std::map<SPOTriplet, Perceptron*>::iterator it=prcps.begin(); it!=prcps.end(); ++it ) {

                q_spap = ( * ( it->second ) ) ( image );
                if ( q_spap > min_q_spap ) {
                    min_q_spap = q_spap;
                }
            }

        return min_q_spap;
    }
#endif

#ifdef FEELINGS
    double max_ap_Q_sp_ap_f ( double image[] ) {
        double q_spap;
        double min_q_spap = -std::numeric_limits<double>::max();

        for ( std::map<Feeling, Perceptron*>::iterator it=prcps_f.begin(); it!=prcps_f.end(); ++it ) {

            q_spap = ( * ( it->second ) ) ( image );
            if ( q_spap > min_q_spap ) {
                min_q_spap = q_spap;
            }
        }

        return min_q_spap;
    }
#endif

#ifdef LZW_TREE
    SPOTriplet argmax_ap_f_lzw ( std::string prg, double image[] ) {
        double min_f = -std::numeric_limits<double>::max();
        SPOTriplet ap;

#ifdef QNN_DEBUG_BREL
        double sum {0.0}, rel;
        double a = std::numeric_limits<double>::max(), b = -std::numeric_limits<double>::max();
#endif

        std::map<SPOTriplet, TripletNode*> children = tree->getChildren();

        int rN = children.size();

        //std::uniform_int_distribution<int> zdist ( 0, rN+1+rN/10 );
        //std::uniform_int_distribution<int> zdist ( 0, 100 );
        //if ( zdist ( zgen ) < rN )
        //if ( rN && zdist ( zgen ) < 95)
        if ( rN ) {
            for ( std::map<SPOTriplet, TripletNode*>::iterator it=children.begin(); it!=children.end(); ++it ) {
                /*
                    for ( std::map<SPOTriplet, Perceptron*>::iterator it=prcps.begin(); it!=prcps.end(); ++it )
                      {
                */
                //double  q_spap = ( * ( it->second ) ) ( image );
                double  q_spap = ( * ( prcps[it->first] ) ) ( image );
                double explor = f ( q_spap, frqs[it->first][prg] );

#ifdef QNN_DEBUG_BREL
                sum += q_spap;

                if ( q_spap > b ) {
                    b = q_spap;
                }

                if ( q_spap < a ) {
                    a = q_spap;
                }
#endif

                if ( explor >= min_f ) {
                    min_f = explor;
                    ap = it->first;
#ifdef QNN_DEBUG_BREL
                    rel = q_spap;
#endif
                }
            }

#ifdef QNN_DEBUG
            if ( b == a ) {
                relevance = 1.0 - ( rel - sum/ ( ( double ) children.size() ) );
            } else {
                relevance = ( rel - sum/ ( ( double ) children.size() ) ) / ( b-a );
            }
#endif

        } else {
            for ( std::map<SPOTriplet, Perceptron*>::iterator it=prcps.begin(); it!=prcps.end(); ++it ) {
                double  q_spap = ( * ( it->second ) ) ( image );
                double explor = f ( q_spap, frqs[it->first][prg] );

#ifdef QNN_DEBUG_BREL
                sum += q_spap;

                if ( q_spap > b ) {
                    b = q_spap;
                }

                if ( q_spap < a ) {
                    a = q_spap;
                }
#endif

                if ( explor >= min_f ) {
                    min_f = explor;
                    ap = it->first;
#ifdef QNN_DEBUG_BREL
                    rel = q_spap;
#endif
                }
            }

#ifdef QNN_DEBUG
            relevance = ( rel - sum/ ( ( double ) prcps.size() ) ) / ( b-a );
#endif

        }


        return ap;
    }
#endif

    SPOTriplet argmax_ap_f ( std::string prg, double image[] ) {
        double min_f = -std::numeric_limits<double>::max();
        SPOTriplet ap;

#ifdef QNN_DEBUG_BREL
        double sum {0.0}, rel;
        double a = std::numeric_limits<double>::max(), b = -std::numeric_limits<double>::max();
#endif

        for ( std::map<SPOTriplet, Perceptron*>::iterator it=prcps.begin(); it!=prcps.end(); ++it ) {

            double  q_spap = ( * ( it->second ) ) ( image );
            double explor = f ( q_spap, frqs[it->first][prg] );

#ifdef QNN_DEBUG_BREL
            sum += q_spap;

            if ( q_spap > b ) {
                b = q_spap;
            }

            if ( q_spap < a ) {
                a = q_spap;
            }
#endif

            if ( explor >= min_f ) {
                min_f = explor;
                ap = it->first;
#ifdef QNN_DEBUG_BREL
                rel = q_spap;
#endif
            }
        }
#ifdef QNN_DEBUG
        relevance = ( rel - sum/ ( ( double ) prcps.size() ) ) / ( b-a );
#endif

        return ap;
    }

#ifdef FEELINGS
    Feeling argmax_ap_f_f ( std::string prg, double image[] ) {
        double min_f = -std::numeric_limits<double>::max();
        Feeling ap;

#ifdef QNN_DEBUG_BREL
        double sum {0.0}, rel;
        double a = std::numeric_limits<double>::max(), b = -std::numeric_limits<double>::max();
#endif

        for ( std::map<Feeling, Perceptron*>::iterator it=prcps_f.begin(); it!=prcps_f.end(); ++it ) {

            double  q_spap = ( * ( it->second ) ) ( image );
            double explor = f ( q_spap, frqs_f[it->first][prg] );

#ifdef QNN_DEBUG_BREL
            sum += q_spap;

            if ( q_spap > b ) {
                b = q_spap;
            }

            if ( q_spap < a ) {
                a = q_spap;
            }
#endif

            if ( explor >= min_f ) {
                min_f = explor;
                ap = it->first;
#ifdef QNN_DEBUG_BREL
                rel = q_spap;
#endif
            }
        }
#ifdef QNN_DEBUG
        relevance_f = ( rel - sum/ ( ( double ) prcps_f.size() ) ) / ( b-a );
#endif

        return ap;
    }
#endif

    SPOTriplet operator() ( SPOTriplet triplet, std::string prg, double image[] ) {

        *this << triplet;

        // Here 'triplet' will also be used as a simplified state in further developments
        // s' = triplet
        // r' = reward

        double reward =
            //3.0 * triplet.cmp ( prev_action ) - 1.5;
            //( triplet == prev_action ) ?1.0:-2.0;
            ( triplet == prev_action ) ?max_reward:min_reward;

#ifdef FEELINGS
        Feeling feeling = prcps_f.begin()->first;
#endif

        if ( prcps.find ( triplet ) == prcps.end() ) {

#ifdef PLACE_VALUE
//        prcps[triplet] = new Perceptron ( 3, 10*3, 4,  1 ); //exp.a1 // 302
            prcps[triplet] = new Perceptron ( 5, 10*3, 16, 8, 4,  1 );

#elif FOUR_TIMES
            prcps[triplet] = new Perceptron ( 3, 2*10*2*80, 32,  1 );

#elif CHARACTER_CONSOLE
            prcps[triplet] = new Perceptron ( 3, 10*80, 32,  1 ); //exp.a1 // 302

            //prcps[triplet] = new Perceptron ( 3, 10*80, 64,  1 ); //exp.a4
            //prcps[triplet] = new Perceptron ( 4, 10*80, 256, 32,  1 );
            //prcps[triplet] = new Perceptron ( 5, 10*80, 256, 128, 32, 1 ); // 355
            //prcps[triplet] = new Perceptron ( 5, 10*80, 196, 32,  32, 1 ); // 302
            //prcps[triplet] = new Perceptron ( 5, 10*80, 400, 400,  32, 1 ); // 302
#elif LIFEOFGAME
            //prcps[triplet] = new Perceptron ( 3, 9, 32, 1 );
//	    prcps[triplet] = new Perceptron ( 4, 2, 64, 9, 1 );
            prcps[triplet] = new Perceptron ( 3, 2, 6, 1 );
#else
            prcps[triplet] = new Perceptron ( 3, 256*256, 80, 1 );
            //prcps[triplet] = new Perceptron ( 3, 256*256, 400, 1 );
#endif
        }

        SPOTriplet action = triplet;

        if ( prev_reward >  -std::numeric_limits<double>::max() ) {
            ++frqs[prev_action][prev_state];
#ifdef FEELINGS
            ++frqs_f[prev_feeling][prev_state];
#endif

#ifndef SARSA
            double max_ap_q_sp_ap = max_ap_Q_sp_ap ( image );
#else
            double max_ap_q_sp_ap = ( *prcps[action] ) ( image );
#endif

#ifdef FEELINGS
            double max_ap_q_sp_ap_f = max_ap_Q_sp_ap_f ( image );
#endif
            double old_q_q_s_a_nn_q_s_a;

            for ( int z {0}; z<10; ++z ) {
                double nn_q_s_a = ( *prcps[prev_action] ) ( prev_image );
#ifdef FEELINGS
                double nn_q_s_a_f = ( *prcps_f[prev_feeling] ) ( prev_image );
#endif

                double q_q_s_a = nn_q_s_a +
                                 alpha ( frqs[prev_action][prev_state] ) *
                                 ( reward + gamma * max_ap_q_sp_ap - nn_q_s_a );

#ifdef FEELINGS
                double q_q_s_a_f = nn_q_s_a_f +
                                   alpha ( frqs_f[prev_feeling][prev_state] ) *
                                   ( reward + gamma * max_ap_q_sp_ap_f - nn_q_s_a_f );
#endif
                prcps[prev_action]->learning ( prev_image, q_q_s_a, nn_q_s_a );

#ifdef FEELINGS
                prcps_f[prev_feeling]->learning ( prev_image, q_q_s_a_f, nn_q_s_a_f );
#endif

#ifdef NN_DEBUG
                std::cerr << "### "
                          << q_q_s_a - nn_q_s_a
                          << " "
                          << q_q_s_a
                          << " "
                          << nn_q_s_a
                          << std::endl;
#ifdef FEELINGS
                std::cerr << "###Feelings "
                          << q_q_s_a_f - nn_q_s_a_f
                          << " "
                          << q_q_s_a_f
                          << " "
                          << nn_q_s_a_f
                          << std::endl;
#endif
#endif

                if ( std::fabs ( old_q_q_s_a_nn_q_s_a - ( q_q_s_a - nn_q_s_a ) ) <= 0.0000000001 ) {
                    break;
                }

                old_q_q_s_a_nn_q_s_a = q_q_s_a - nn_q_s_a;

            }


//        action = argmax_ap_f ( prg, image );
#ifdef LZW_TREE
            action = argmax_ap_f_lzw ( prg, image );
#else
            action = argmax_ap_f ( prg, image );
#endif

#ifdef FEELINGS
            feeling = argmax_ap_f_f ( prg, image );
#endif
        }

        prev_state = prg; 		// s <- s'
        prev_reward = reward;	// r <- r'
        prev_action = action;	// a <- a'
#ifdef FEELINGS
        prev_feeling = feeling;	// a <- a'
#endif

#ifdef PLACE_VALUE
        std::memcpy ( prev_image, image, 10*3*sizeof ( double ) );
#elif FOUR_TIMES
        std::memcpy ( prev_image, image, 2*10*2*80*sizeof ( double ) );
#elif CHARACTER_CONSOLE
        std::memcpy ( prev_image, image, 10*80*sizeof ( double ) );
#elif LIFEOFGAME
        std::memcpy ( prev_image, image, 40*sizeof ( double ) );



#else
        std::memcpy ( prev_image, image, 256*256*sizeof ( double ) );
#endif
        return action;
    }

#else

//    double max_ap_Q_sp_ap ( std::string prg ) {
    double max_ap_Q_sp_ap ( int prg ) {
        double q_spap;
        double min_q_spap = -std::numeric_limits<double>::max();

        for ( std::map<int, std::map<int, double>>::iterator it=table_.begin(); it!=table_.end(); ++it ) {
            q_spap = it->second[prg];
            if ( q_spap > min_q_spap ) {
                min_q_spap = q_spap;
            }
        }

        return min_q_spap;
    }

    int argmax_ap_f ( int prg ) {
        double q_spap;
        double min_f = -std::numeric_limits<double>::max();
        int ap;

        for ( std::map<int, std::map<int, double>>::iterator it=table_.begin(); it!=table_.end(); ++it ) {

            q_spap = it->second[prg];

            double explor = f ( q_spap, frqs[it->first][prg] );

            if ( explor > min_f ) {
                min_f = explor;
                ap = it->first;
            }
        }

        return ap;
    }

    SPOTriplet exec ( int toi, int * center_of_tape, int noc, int noc2 ) {

        /*
          int b[3];

          b[0] = center_of_tape[noc-1];
          b[1] = center_of_tape[noc];
          b[2] = center_of_tape[noc+1];

          //state = rules.to[toi][0];
          //tape.set_tape ( tape.tapei, rules.to[toi][1] );
          int tapei =  1;
          b[tapei] = tr.to[toi][1];
          tapei += ( tr.to[toi][2] - 1 );

          return tr.to[toi][0] + 100 * b[tapei];
        */


        int tapei =  noc;
        center_of_tape[tapei] = tr.to[toi][1];
        tapei += ( tr.to[toi][2] - 1 );

        center_of_tape[tapei] = tr.to[toi][0] + 100 * center_of_tape[tapei];
        /*
            return std::make_tuple(center_of_tape[tapei-4], center_of_tape[tapei-3], center_of_tape[tapei-2], center_of_tape[tapei-1],
                    center_of_tape[tapei],
                    center_of_tape[tapei+1], center_of_tape[tapei+2], center_of_tape[tapei+3], center_of_tape[tapei+4]);
        */

        SPOTriplet config;
        config.reserve ( 1000 );
        for ( int i {-noc2}; i<= noc2; ++i ) {
            config.push_back ( center_of_tape[tapei+i] );
        }

        return config;

    }

    TransitionRules<5> tr;

// cc*


    int prev_center[2*1000+1];


    int operator() ( SPOTriplet triplet, int * center_of_tape, int noc, int noc2 ) {

        static int c = 0;

        SPOTriplet action = triplet;
        int rc {0};

        if ( c>0 ) {

            //auto prg = triplet[noc2];

            // s' = triplet
            // r' = reward

            int buf[2*1000+1];
            //std::vector<SPOTriplet> confs;


            for ( int i {0}; i<5*2*3; ++i ) {


                std::memcpy ( buf, prev_center, ( 2*1000+1 ) *sizeof ( int ) );

                action = exec ( i, buf, noc, noc2 );
                //if ( action == prev_state ) {
		if ( action == triplet ) {
                    ++rc;
                }

            }

            if ( rc == 0 ) {
                std::cout << rc << "-0000000000" << std::endl;
            } else if ( rc == 1 ) {
                std::cout << rc << "-1111111111" << std::endl;
            } else {
                std::cout << rc << "-2222222222" << std::endl;
		
	//	std::cout << 
		
		
		
            }
        }

        ++c;
        std::memcpy ( prev_center, center_of_tape, ( 2*1000+1 ) *sizeof ( int ) );

        prev_state = triplet; 		// s <- s'

        if ( rc >1 ) {
	  c = 0;
            return -rc;
        } else {
            return 0;
        }



    }



    SPOTriplet operator() ( SPOTriplet triplet, int * center_of_tape, int noc, int noc2, /*std::string prg,*/ bool isLearning ) {

        auto prg = triplet[noc2];

        // s' = triplet
        // r' = reward


//        if ( prg ==0 ) {
        std::cout << " QQQ\n";

        for ( auto i:triplet ) {
            std::cout  << i << " ";
        }
        std::cout << " PPP\n";

        for ( auto i:prev_action ) {
            std::cout  << i << " ";
        }
//        }

        double reward =
            //3.0*triplet.cmp ( prev_action ) - 1.5;
            ( triplet == prev_action ) ?max_reward:min_reward;

        /*
                if ( triplet == prev_action ) {
                    reinforced_action.first = prev_state;
                    reinforced_action.second = prev_action;

        	    ++rules[reinforced_action];

        	}
        */

        if ( triplet == prev_action ) {

            //reinforced_action.first = prev_state;
            if ( prev_state[noc2] > 50 ) {
                reinforced_action.first = ( prev_state[noc2] -100 ) *2+1;
            } else {
                reinforced_action.first = prev_state[noc2] *2;
            }

            reinforced_action.second = prev_action2;

            ++rules[reinforced_action];

        }

        SPOTriplet action = triplet;
        //SPOTriplet action2 = 0;
        int action2 = -1;

        if ( prev_reward >  -std::numeric_limits<double>::max() ) {

            isLearning = true;

            if ( isLearning ) {


                /*
                            if ( triplet == prev_action ) {
                                //reinforced_action.first = prev_state;
                 if(prev_state > 50)
                   reinforced_action.first =( prev_state-100)*2+1;
                 else
                   reinforced_action.first =prev_state*2;

                                reinforced_action.second = prev_action2;

                                ++rules[reinforced_action];


                            }
                */


                ++frqs[prev_action2][prev_state[noc2]];

                /*
                        for ( int i {0}; i<5*2*3; ++i ) {
                            table_[i][prg] = table_[i][prg];
                        }
                */

                double max_ap_q_sp_ap = max_ap_Q_sp_ap ( prg );

                table_[prev_action2][ prev_state[noc2]] =
                    table_[prev_action2][prev_state[noc2]] +
                    alpha ( frqs[prev_action2][ prev_state[noc2]] ) *
                    ( reward + gamma * max_ap_q_sp_ap - table_[prev_action2][ prev_state[noc2]] );
            }

            action2 = argmax_ap_f ( prg );
            if ( action2 != -1 && action2 != 99 ) {
                action = exec ( action2, center_of_tape, noc, noc2 );
            }

            if ( prg ==0 ) {
                std::cout << " YYY" << action2 <<std::endl;
            }


        }

        //prev_state = prg; 		// s <- s'
        prev_state = triplet; 		// s <- s'
        prev_reward = reward;   	// r <- r'
        prev_action = action;	// a <- a'
        prev_action2 = action2;

        return action;
    }

#endif

    double reward ( void ) {
        return prev_reward;
    }

    SPOTriplet action ( void ) {
        return prev_action;
    }

#ifdef FEELINGS
    Feeling feeling ( void ) {
        return prev_feeling;
    }
#endif
    double alpha ( int n ) {

        return ( ( double ) n ) / ( ( ( double ) n ) + 100.0 );

    }

    void clearn ( void ) {

        for ( std::map<int, std::map<int, int>>::iterator it=frqs.begin(); it!=frqs.end(); ++it ) {

            for ( std::map<int, int>::iterator itt=it->second.begin(); itt!=it->second.end(); ++itt ) {
                itt->second = 0;
            }
        }

    }

    void clear ( void ) {
        tree = &root;
        depth = 0;
    }

    void debug_tree ( void ) {
        int save_depth = depth;
        depth = 0;
        //debug_tree ( &root, std::cerr );
        depth = save_depth;
    }

    double sigmoid ( int n ) {
        return 1.0/ ( 1.0 + exp ( -n ) );
    }

    void scalen ( double s ) {

        for ( std::map<int, std::map<int, int>>::iterator it=frqs.begin(); it!=frqs.end(); ++it ) {

            for ( std::map<int, int>::iterator itt=it->second.begin(); itt!=it->second.end(); ++itt ) {
                //itt->second -= ( itt->second / 5 );
                itt->second *= s;
            }
        }

    }
    /*
        void save_prcps ( std::fstream & samuFile ) {
            samuFile << prcps.size();

            int prev_p {0};
            for ( std::map<SPOTriplet, Perceptron*>::iterator it=prcps.begin(); it!=prcps.end(); ++it ) {
                int p = ( std::distance ( prcps.begin(), it ) * 100 ) / prcps.size();
                if ( p > prev_p+9 ) {
                    std::cerr << "Saving Samu: "
                              << p
                              << "% (perceptrons)"
                              << std::endl;
                    prev_p = p;
                }
                samuFile << " "
                         << it->first;
                it->second->save ( samuFile );
            }
        }

        void save_frqs ( std::fstream & samuFile ) {
            samuFile << std::endl
                     << frqs.size();

            int prev_p {0};
            for ( std::map<SPOTriplet, std::map<std::string, int>>::iterator it=frqs.begin(); it!=frqs.end(); ++it ) {

                int p = ( std::distance ( frqs.begin(), it ) * 100 ) / frqs.size();
                if ( p > prev_p+9 ) {
                    std::cerr << "Saving Samu: "
                              << p
                              << "% (frequency table)"
                              << std::endl;
                    prev_p = p;
                }

                samuFile << " "
                         << it->first
                         << " "
                         << it->second.size();
                for ( std::map<std::string, int>::iterator itt=it->second.begin(); itt!=it->second.end(); ++itt ) {
                    samuFile << " "
                             << itt->first
                             << " "
                             << itt->second;
                }
            }

        }

        void save ( std::string & fname ) {
            std::fstream samuFile ( fname,  std::ios_base::out );

            save_prcps ( samuFile );
            save_frqs ( samuFile );

            samuFile.close();
        }

        void load_prcps ( std::fstream & file ) {
            int prcpsSize {0};
            file >> prcpsSize;

            int prev_p {0};
            SPOTriplet t;
            for ( int s {0}; s< prcpsSize; ++s ) {
                int p = ( s * 100 ) / prcpsSize;
                if ( p > prev_p+9 ) {
                    std::cerr << "Loading Samu: "
                              << p
                              << "% (perceptrons)"
                              << std::endl;
                    prev_p = p;
                }


                file >> t;

                prcps[t] = new Perceptron ( file );
            }

        }

        void load_frqs ( std::fstream & file ) {
            int frqsSize {0};
            file >> frqsSize;

            int prev_pc {0};
            int mapSize {0};
            SPOTriplet t;
            std::string p;
            int n;
            for ( int s {0}; s< frqsSize; ++s ) {

                int pc = ( s * 100 ) / frqsSize;
                if ( pc > prev_pc+9 ) {
                    std::cerr << "Loading Samu: "
                              << pc
                              << "% (frequency table)"
                              << std::endl;
                    prev_pc = pc;
                }

                file >> t;
                file >> mapSize;
                for ( int ss {0}; ss< mapSize; ++ss ) {
                    file >> p;
                    file >> n;

                    frqs[t][p] = n;
                }
            }
        }


        void load ( std::fstream & file ) {
            load_prcps ( file );
            load_frqs ( file );
        }
    */
    int get_N_e ( void ) const {
        return N_e;
    }

    void set_N_e ( int N_e ) {
        this->N_e = N_e;
    }

    double get_max_reward ( void ) const {
        return max_reward;
    }

    double get_min_reward ( void ) const {
        return min_reward;
    }

    void operator<< ( SPOTriplet triplet ) {
        TripletNode *p = tree->getChild ( triplet );
        if ( !p ) {
            if ( depth < 10 ) {
                TripletNode *tn = new TripletNode ( triplet );
                tree->setChild ( triplet, tn );
                tree = &root;
                depth = 0;
            } else {
                tree = &root;
                depth = 0;
                *this << triplet;
            }
        } else {
            tree = p;
            ++depth;
        }
    }

    ReinforcedAction reinforcedAction() const {
        return reinforced_action;
    }

    int getNumRules() const {
        return rules.size();
    }

    std::string printRules() {

        std::stringstream ss;

        ss << rules.size();

        for ( auto& rule : rules ) {
            ss << ", " <<rule.first.first <<","  << rule.first.second << "(" << rule.second<< ") ";
        }


        return ss.str();

    }

    std::string multTos ( std::map<int, std::vector<int>> &machines ) {
        std::stringstream ss;
        int m = 1;
        for ( auto& mm : machines ) {
            m *= mm.second.size();
            ss << "*" << mm.second.size();

        }
        ss << "=" << m;

        return ss.str();
    }

    int frek ( int from, int to ) {
        for ( auto& rule : rules ) {
            if ( rule.first.first == from && rule.first.second == to ) {
                return rule.second;
            }
        }

        return 0;

    }

    std::string printMachines() {

        std::stringstream ss;

        std::map<int, std::vector<int>> machines;
        std::map<int, int> maxs;

        for ( auto& rule : rules ) {
            if ( rule.second > 10*N_e && rule.first.first >= 0 ) {
                machines[rule.first.first].push_back ( rule.first.second );
                if ( rule.second > maxs[rule.first.first] ) {
                    maxs[rule.first.first] = rule.second;
                }
            }
        }

        ss << multTos ( machines );
        for ( auto& mm : machines ) {
            //int max = (4*maxs[mm.first])/5;
            int max = maxs[mm.first]/10;

            ss << "[" << max << "]";

            mm.second.erase ( std::remove_if (
                                  mm.second.begin(), mm.second.end(),
            [=] ( int & n ) {
                return frek ( mm.first, n ) < max;
            } ), mm.second.end() );

        }
        ss << multTos ( machines );

        std::vector<std::vector<std::pair<int, int>>> oldtms;
        std::vector<std::vector<std::pair<int, int>>> newtms;

        for ( auto& m : machines ) {

            newtms.clear();

            if ( oldtms.size() > 0 )
                for ( auto& mm : oldtms ) {

                    for ( auto& tos : m.second ) {

                        std::vector<std::pair<int, int>> mmm = mm;
                        std::pair<int, int> p = std::make_pair ( m.first, tos );
                        mmm.push_back ( p );
                        newtms.push_back ( mmm );


                    }

                }
            else {

                for ( auto& tos : m.second ) {

                    std::vector<std::pair<int, int>> mm;
                    std::pair<int, int> p = std::make_pair ( m.first, tos );
                    mm.push_back ( p );
                    newtms.push_back ( mm );

                }

            }

            oldtms = newtms;

        }

        ss << "Possible TMs:" << oldtms.size() << std::endl;

        for ( auto& mm : oldtms ) {

            ss << "***TM: " << mm.size();
            for ( auto& mmm : mm ) {

                ss << ", " << mmm.first << ", " << mmm.second;

            }
            ss << std::endl;

        }

        return ss.str();

    }

    std::string printSortedRules() {

        std::vector<std::pair<std::pair<int, int>, int>> tmp;

        for ( auto& rule : rules ) {
            std::pair<std::pair<int, int>, int> p {{rule.first.first, rule.first.second}, rule.second};
            tmp.push_back ( p );
        }

        std::sort (
            std::begin ( tmp ), std::end ( tmp ),
        [=] ( auto&& t1, auto&&t2 ) {
            return t1.second > t2.second;
        }
        );

        std::stringstream ss;

        ss << tmp.size();

        for ( auto& rule : tmp ) {
            //ss << ", " <<rule.first.first <<","  << rule.first.second << "(" << rule.second<< ") ";
            ss << ", " <<rule.first.first <<", "  << rule.first.second;

        }
        return ss.str();

    }


private:

    class TripletNode
    {
    public:
        TripletNode ( ) {
        };
        TripletNode ( SPOTriplet triplet ) :triplet ( triplet ) {
        };
        ~TripletNode () {
        };
        void setChild ( SPOTriplet &triplet, TripletNode * newChild ) {
            children[triplet] = newChild;
        }
        TripletNode  *getChild ( SPOTriplet &triplet ) const {
            std::map<SPOTriplet, TripletNode*>::const_iterator it = children.find ( triplet );

            if ( it != children.end() ) {
                return ( *it ).second;
            } else {
                return nullptr;
            }
        }
        std::map<SPOTriplet, TripletNode*> & getChildren () {
            return children;
        }
        SPOTriplet getTriplet () const {
            return triplet;
        }

    private:
        TripletNode ( const TripletNode & );
        TripletNode & operator= ( const TripletNode & );
        SPOTriplet triplet;
        std::map<SPOTriplet, TripletNode*> children;
    };

    TripletNode root;
    TripletNode *tree;
    int depth {0};

    /*
    std::random_device zinit;
    std::default_random_engine zgen {zinit() };
    */
    /*
        void debug_tree ( TripletNode * node, std::ostream & os ) {
            if ( node != nullptr ) {
                ++depth;
                std::map<SPOTriplet, TripletNode*> children = node->getChildren();

                for ( std::map<SPOTriplet, TripletNode*>::iterator it=children.begin(); it!=children.end(); ++it ) {
                    debug_tree ( ( *it ).second, os );
                }

                for ( int i {0}; i < depth; ++i )
                    if ( i == ( 2* ( depth-1 ) ) /2 ) {
                        os  << depth - 1;
                    } else {
                        os << "__";
                    }

                os << "__ "
                   << node->getTriplet ()
                   << std::endl;
                --depth;
            }
        }
    */
    int N_e = 3*5*2*3;

    QL ( const QL & );
    QL & operator= ( const QL & );

#ifdef Q_LOOKUP_TABLE
    double gamma = .5;
#else
    double gamma = .5;
#endif

#ifdef Q_LOOKUP_TABLE
    //std::map<SPOTriplet, std::map<std::string, double>> table_;
    std::map<int, std::map<int, double>> table_;
#else
    std::map<SPOTriplet, Perceptron*> prcps;
#ifdef FEELINGS
    std::map<Feeling, Perceptron*> prcps_f;
#endif
#ifdef QNN_DEBUG
    double relevance {0.0};
#ifdef FEELINGS
    double relevance_f {0.0};
#endif
#endif
#endif

    //std::map<SPOTriplet, std::map<std::string, int>> frqs;
    std::map<int, std::map<int, int>> frqs;

#ifdef FEELINGS
    std::map<Feeling, std::map<std::string, int>> frqs_f;
#endif
    SPOTriplet prev_action;
    int prev_action2;
#ifdef FEELINGS
    Feeling prev_feeling {"Hello, World!"};
#endif
    //std::string prev_state;
    SPOTriplet prev_state;

    double prev_reward { -std::numeric_limits<double>::max() };

    /*
    double max_reward {13.1};
    double min_reward {-3.1};
    */

    double max_reward {0.0};
    double min_reward {-1000000000.00};

#ifdef PLACE_VALUE
    double prev_image [10*3];
#elif FOUR_TIMES
    double prev_image [2*10*2*80];
#elif CHARACTER_CONSOLE
    double prev_image [10*80];
#elif LIFEOFGAME
    double prev_image [40];
#else
    double prev_image [256*256];
#endif

    //ReinforcedAction reinforced_action {"unreinforced", -1};
    ReinforcedAction reinforced_action {-1000, -1000};
    //std::map<ReinforcedAction, int> rules;
    std::map<ReinforcedAction, int> rules;

};

#endif


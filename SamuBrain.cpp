/**
 * @brief Samu has learnt the rules of Conway's Game of Life
 *
 * @file GameOfLife.h
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
 * SamuBrain, exp. 4, cognitive mental organs: MPU (Mental Processing Unit), Q-- lerning, acquiring higher-order knowledge
 *
 * This is an example of the paper entitled "Samu in his prenatal development".
 *
 * Previous experiments
 *
 * Samu (Nahshon)
 * http://arxiv.org/abs/1511.02889
 * https://github.com/nbatfai/nahshon
 *
 * SamuLife
 * https://github.com/nbatfai/SamuLife
 * https://youtu.be/b60m__3I-UM
 *
 * SamuMovie
 * https://github.com/nbatfai/SamuMovie
 * https://youtu.be/XOPORbI1hz4
 *
 * SamuStroop
 * https://github.com/nbatfai/SamuStroop
 * https://youtu.be/6elIla_bIrw
 * https://youtu.be/VujHHeYuzIk
 */


#include "SamuBrain.h"

SamuBrain::SamuBrain ( int w, int h ) : m_w ( w ), m_h ( h )
{
        m_morgan = newMPU();

        m_searching = false;
	
	config.reserve(100);
}

SamuBrain::~SamuBrain()
{

        for ( auto& mpu : m_brain ) {
                delete mpu.second;
        }

}

MentalProcessingUnit::MentalProcessingUnit ( int w, int h ) : m_w ( w ), m_h ( h )
{

        m_samuQl = new QL*[m_h];

        for ( int i {0}; i<m_h; ++i ) {
                m_samuQl[i] = new QL [m_w];
        }

        m_prev = new int*[m_h];
        fp = new int*[m_h];
        fr = new int*[m_h];

        for ( int i {0}; i<m_h; ++i ) {
                m_prev[i] = new int [m_w];
                fp[i] = new int [m_w];
                fr[i] = new int [m_w];
        }

        for ( int r {0}; r<m_h; ++r )
                for ( int c {0}; c<m_w; ++c ) {
                        fr[r][c] =fp[r][c] = m_prev[r][c] = 0;
                }

}

void MentalProcessingUnit::cls ( )
{
        for ( int r {0}; r<m_h; ++r )
                for ( int c {0}; c<m_w; ++c ) {
                        fr[r][c] =fp[r][c] = m_prev[r][c] = 0;
                }
}

MentalProcessingUnit::~MentalProcessingUnit ( )
{

        for ( int i {0}; i<m_h; ++i ) {
                delete[] m_samuQl[i];
        }

        delete[] m_samuQl;

        for ( int i {0}; i<m_h; ++i ) {
                delete [] m_prev[i];
        }
        delete [] m_prev;

}


MORGAN SamuBrain::newMPU ()
{

        MORGAN morgan = new MentalProcessingUnit ( m_w, m_h );

        std::stringstream ss;
        ss << "Foobar";
        ss << m_brain.size();
        ss << " " << morgan->getSamu();
        std::string mpuName = ss.str();

        m_brain[mpuName] = morgan;

        return morgan;
}

int SamuBrain::getW() const
{
        return m_w;
}
int SamuBrain::getH() const
{
        return m_h;
}

bool SamuBrain::isSearching() const
{
        return m_searching;
}

/*
bool SamuBrain::hasAlreadyLearned() const
{
  return haveAlreadyLearnt;
}
*/

int SamuBrain::nofMPUs() const
{
        return m_brain.size();
}

/*
double SamuBrain::howMuchLearned() const
{
  return howMuchLearned ( samuQl );
}

double SamuBrain::howMuchLearned ( MPU samuQl ) const
{
  int s {0};
  for ( int r {0}; r<m_h; ++r )
    {
      for ( int c {0}; c<m_w; ++c )
        {
          if ( samuQl[r][c].reward() == samuQl[r][c].get_max_reward() )
            {
              ++s;
            }

        }
    }

  return ( ( double ) s ) / ( ( double ) m_h*m_w );
}
*/

int SamuBrain::pred ( int **reality, int * center_of_tape, int noc, int **predictions, int isLearning, int & vsum, int & sum2, int & vsum2 )
{
        return pred ( m_morgan, reality, center_of_tape, noc, predictions, isLearning, vsum, sum2, vsum2 );
}

/*
int SamuBrain::pred ( MORGAN morgan, int **reality, int **predictions, int isLearning, int & vsum )
{

  MPU samuQl = morgan->getSamu();
  int ** prev = morgan->getPrev();

  //double img_input[40];
  int colors[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  int sum {0};

  vsum = 0;

  for ( int r {0}; r<m_h; ++r )
    {
      for ( int c {0}; c<m_w; ++c )
        {

          std::stringstream ss;
          int ii {0};

          for ( int ci {0}; ci<5; ++ci )
            {
              colors[ci] = 0;
            }

          for ( int i {-1}; i<2; ++i )
            for ( int j {-1}; j<2; ++j )

              if ( ! ( ( i==0 ) && ( j==0 ) ) )

                {
                  int o = c + j;
                  if ( o < 0 )
                    {
                      o = m_w-1;
                    }
                  else if ( o >= m_w )
                    {
                      o = 0;
                    }

                  int s = r + i;
                  if ( s < 0 )
                    {
                      s = m_h-1;
                    }
                  else if ( s >= m_h )
                    {
                      s = 0;
                    }

                  ++colors[reality[s][o]];


                } // if

          ss << reality[r][c];
          ss << '|';
          ss << colors[0]; //img_input[1];
          ss << '|';
          ss << colors[1];
          ss << '|';
          ss << colors[2];
          ss << '|';
          ss << colors[3];
          ss << '|';
          ss << colors[4];

          std::string prg = ss.str();

          // with NNs
          //SPOTriplet response = samuQl[r][c] ( lattice[r][c], prg, img_input );
          // without
          SPOTriplet response = samuQl[r][c] ( reality[r][c], prg, isLearning == 0 );

          predictions[r][c] = response;

          if ( ( predictions[r][c] == prev[r][c] ) && ( prev[r][c] != 0 ) )
            {
              ++vsum;
              if ( samuQl[r][c].reward() == samuQl[r][c].get_max_reward() )
                {
                  ++sum;
                }
            }

          prev[r][c] = reality[r][c];

          if ( isLearning>0 && predictions[r][c] == 0 )
            {
              predictions[r][c] = isLearning;
            }

        }

    }

  return sum;
}
*/

int SamuBrain::pred ( MORGAN morgan, int **reality, int * center_of_tape, int noc, int **predictions, int isLearning, int & vsum, int & sum2, int & vsum2 )
{

        MPU samuQl = morgan->getSamu();
        int ** prev = morgan->getPrev();
        int ** fp = morgan->getFp();
        int ** fr = morgan->getFr();

        //double img_input[40];
        int colors[256] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        int sum {0};

        vsum = sum2 = vsum2 = 0;

        /*
        for ( int r {0}; r<m_h; ++r )
          {
            for ( int c {0}; c<m_w; ++c )
              {
        */

        int r = 0;
        int c = 10;


        //  std::stringstream ss;
        //int ii {0};

        //for ( int ci {0}; ci<5; ++ci )
        //  {
        //    colors[ci] = 0;
        //  }

        /*
        for ( int i {-1}; i<2; ++i )
        for ( int j {-1}; j<2; ++j )

          if ( ! ( ( i==0 ) && ( j==0 ) ) )

            {
              int o = c + j;
              if ( o < 0 )
                {
                  o = m_w-1;
                }
              else if ( o >= m_w )
                {
                  o = 0;
                }

              int s = r + i;
              if ( s < 0 )
                {
                  s = m_h-1;
                }
              else if ( s >= m_h )
                {
                  s = 0;
                }

              ++colors[reality[s][o]];


            } // if
        */

        /*
          ss << reality[r][c];
          ss << '|';
          ss << colors[0]; //img_input[1];
          ss << '|';
          ss << colors[1];
          ss << '|';
          ss << colors[2];
          ss << '|';
          ss << colors[3];
          ss << '|';
          ss << colors[4];
        */

        //      ss << reality[r][c];

        //    std::string prg = ss.str();

        // with NNs
        //SPOTriplet response = samuQl[r][c] ( lattice[r][c], prg, img_input );
        // without

        //  prev[r][c] = samuQl[r][c].action();// mintha a samuQl hívása után a predikciót mentettem volna el (B)
        //predictions[r][c] =  prev[r][c];

        //SPOTriplet response = samuQl[r][c] ( reality[r][c], center_of_tape, noc, /*prg,*/ isLearning == 0 );

        /*
        auto state = std::make_tuple ( reality[r][noc-3], reality[r][noc-2],reality[r][noc-1],
                                       reality[r][noc],reality[r][noc+1], reality[r][noc+2], reality[r][noc+3] );
*/

/*	
        auto state = std::make_tuple ( reality[r][noc-4], reality[r][noc-3], reality[r][noc-2],reality[r][noc-1],
                                       reality[r][noc],reality[r][noc+1], reality[r][noc+2], reality[r][noc+3], reality[r][noc+4] );
*/	
	config.clear();
	for(int i{-4}; i<= 4; ++i)
	  config.push_back(reality[r][noc+i]);
	
	
        SPOTriplet response = samuQl[r][c] ( config, center_of_tape, noc, 4, /*prg,*/ isLearning == 0 );


//          if ( prev[r][c] )
        if ( prev[r][c] && reality[r][c] )
                //if ( ( predictions[r][c] == reality[r][c] ) && ( reality[r][c] != 0 ) )
        {
                ++vsum2;
                //if (  samuQl[r][c].reward() == samuQl[r][c].get_max_reward()/*reality[r][c] == prev[r][c]*/ )
                if ( reality[r][c] == prev[r][c] ) {
                        ++sum2;
//		  if(!isLearning)
//		  ++fp[r][c];
                }
        }

        if ( reality[r][c] )
//          if ( prev[r][c] && reality[r][c])
                //if ( ( predictions[r][c] == reality[r][c] ) && ( reality[r][c] != 0 ) )
        {
                ++vsum;
                //if (  samuQl[r][c].reward() == samuQl[r][c].get_max_reward()/*reality[r][c] == prev[r][c]*/ )
                if ( reality[r][c] == prev[r][c] ) {
                        ++sum;
//		  if(!isLearning)
//		  ++fp[r][c];
                }
        }

        // if ( !isLearning )
        {

                if ( reality[r][c] == prev[r][c] ) {
                        if ( fp[r][c] < 255-60 ) {
                                fp[r][c]+=60;
                        }
                } else {
                        if ( fp[r][c] > 60 ) {
                                fp[r][c]-=60;
                        }
                }


                fr[r][c] = samuQl[r][c].getNumRules();

        }

        //prev[r][c] = reality[r][c];




        for ( int c {0}; c<2*10+1; ++c ) {
                predictions[r][c] = prev[r][c];
                prev[r][c] = center_of_tape[c]; // /*predictions[r][c] =*/ response;
        }

        std::cout  << "\nP> *TM: " << samuQl[r][c].printRules().c_str() << std::endl;
        std::cout  << "P> **TM: (sort) " << samuQl[r][c].printSortedRules().c_str() << std::endl;
        std::string pm = samuQl[r][c].printMachines();
        std::cout  << "P> ***TM: " << pm.c_str() << std::endl;

//	   std::string tm314 ("9, 0, 11, 1, 15, 2, 17, 3, 11, 4, 23, 5, 24, 6, 3, 7, 21, 9, 0");
//

        std::string tm314 ( "9, 0, 9, 1, 11, 2, 5, 3, 20, 4, 17, 5, 24, 7, 29, 8, 15, 9, 1" );
        //std::string tm314 ( "9, 0, 11, 1, 15, 2, 17, 3, 11, 4, 23, 5, 24, 6, 3, 7, 21, 9, 0" );
        //std::string tm314 ( "9, 0, 11, 2, 15, 3, 17, 4, 26, 5, 18, 6, 15, 7, 6, 8, 23, 9, 5" );
        //std::string tm314 ( "9, 0, 11, 1, 15, 2, 0, 3, 18, 4, 3, 6, 9, 7, 29, 8, 20, 9, 8" );
	//std::string tm314 ( "9, 0, 11, 1, 12, 2, 17, 3, 23, 4, 3, 5, 8, 6, 26, 8, 15, 9, 5");
        //std::string tm314 ( "9, 0, 9, 1, 12, 2, 15, 3, 21, 4, 29, 5, 1, 7, 24, 8, 2, 9, 27");
	//std::string tm314 ( "9, 0, 21, 1, 9, 2, 24, 3, 6, 4, 3, 5, 20, 6, 17, 7, 0, 9, 15");

	//std::string tm314 ( "9, 0, 9, 1, 11, 2, 17, 3, 21, 4, 19, 5, 29, 6, 5, 7, 6, 8, 8");	
	//std::string tm314 ( "9, 0, 9, 1, 11, 2, 15, 3, 20, 4, 21, 5, 27, 6, 4, 7, 2, 8, 12");
	
        std::size_t found = pm.find ( tm314 );
        if ( found!=std::string::npos ) {
                std::cout  << " +++++++++++ VAN ++++++++++++++++++++++++++++++++++++++++++"<< std::endl;
        } else {
                std::cout  << " ----------- NINCS +++++++++++++++++++++++++++++++++++++"<< std::endl;
        }


        // aligning to psamu1 paper // if ( ( predictions[r][c] == prev[r][c] ) && ( prev[r][c] != 0 ) )
        /*
        	  if ( ( reality[r][c] == prev[r][c] ) && ( prev[r][c] != 0 ) )
        	  //if ( ( predictions[r][c] == reality[r][c] ) && ( reality[r][c] != 0 ) )
                    {
                      ++vsum;
                      if ( samuQl[r][c].reward() == samuQl[r][c].get_max_reward() )
                        {
                          ++sum;
                        }
                    }
        */
        // aligning to psamu1 paper // prev[r][c] = reality[r][c];
        // prev[r][c] = predictions[r][c];
        /*!!!
                  if ( isLearning>0 && predictions[r][c] == 0 )
                    {
                      predictions[r][c] = isLearning;
                    }
        */
//        }

//    }

        return sum;
}

/*
bool SamuBrain::is_habituation ( int q, int w, int e, int r, int t, int z, int &mon )
{
  static int mem {0};
  static int err {0}; // from these this function is not a reentrant one!!!

  const int convergence_limit{4000};
  const int error_limit{3};

  if ( q != 0
       && q == w
       && e != 0
       && e == r
       && t == z
     )
    {

      mon = ( ( double ) mem/ ( double ) convergence_limit ) * 100;
      err = 0;

      if ( mem >= convergence_limit )
        {
          return true;
        }
      else
        {
          ++mem;
          return false;
        }
    }
  else
    {

      if ( q == 0
           && q == w
           && e != 0
           && e == r
           && t == z
           && err < error_limit )
        {
          ++err;
        }
      else
        {
          err = mon = mem = 0;
        }
      return false;
    }
}
*/

bool Habituation::is_newinput ( int vsum, int sum ) //, double &mon )
{
// return ( sum < masum ) || ( vsum < mavsum );
        return (
                       ( vsum != sum ) && ( ( sum < masum ) || ( vsum < mavsum ) || ( ssum +1 < sum10 ) )
               )
               ||
               ( vsum == sum && ( ( mavsum - asum[ma_limit-1] ) >= 4 ) );

}

bool Habituation::is_habituation ( int vsum, int sum, double &mon )
{

        //int ssum {0};
        ssum = 0;
        int svsum {0};

        for ( int ci {0}; ci<ma_limit-1; ++ci ) {
                ssum += ( msum[ci] = msum[ci+1] );
                svsum += ( asum[ci] = asum[ci+1] );
        }
        ssum += ( msum[ma_limit-1] = sum );
        svsum += ( asum[ma_limit-1] = vsum );

        masum = ( int ) ( ( double ) ssum / ( double ) ma_limit );
        mavsum = ( int ) ( ( double ) svsum / ( double ) ma_limit );

        int q, w,  e,  r, t, z;

        q= vsum, w =sum;
        e= mavsum;
        r= masum;
        t= masum - msum[ma_limit-1];
        z = mavsum - asum[ma_limit-1];

        if ( ( ++counter ) % 15 == 0 )
                sum10	= ssum;

        /*
          std::cout  << "   HABITUATION MONITOR:"
                   << "(isHABI MPU)"
                   << vsum << sum << mavsum << masum
                   << masum - msum[ma_limit-1]
                   << mavsum - asum[ma_limit-1]
                   << "|||" << sum10 << ssum;
        */
        if ( q != 0
                        && q == w
                        /*
                               && e != 0
                               && e == r
                               && t == z*/
           ) {

                mon = ( ( double ) mem/ ( double ) convergence_limit );
                err = 0;

                if ( mem >= convergence_limit ) {
                        return true;
                } else {
                        ++mem;
                        return false;
                }
        } else {
                /*
                      if ( q == 0
                           && q == w
                	 )
                	 */
                if ( q == 0
                                && q == w
                                && e != 0
                                && e == r
                                && t == z
                   ) {
                        ++err;
                        //  if(mem >1) mem-=2;
                        mon = ( ( double ) mem/ ( double ) convergence_limit );

                        if ( mem >= convergence_limit ) {
                                ;
                        } else {
                                ;//  ++mem;
                        }
                } else {
                        if ( err < error_limit ) {
                                ++err;
                        } else {
                                mon = 0.0;
                                err = mem = 0;
                        }
                }
                return false;
        }
}


void SamuBrain::learning ( int **reality, int * center_of_tape, int noc , int **predictions, int ***fp, int ***fr )
{
        this->fp = fp;
        this->fr = fr;

        ++m_internal_clock;

        int sum {0};
        int vsum {0};
        int sum2 {0};
        int vsum2 {0};

        if ( m_searching ) {

                * ( this->fp ) = nullptr ;
                * ( this->fr ) = nullptr ;

                MORGAN maxSamuQl {nullptr};

                int ell = 0	;


                for ( auto& mpu : m_brain ) {

                        MORGAN morgan = mpu.second;

                        sum = pred ( morgan, reality, center_of_tape, noc, predictions, 4, vsum, sum2, vsum2 );

                        double mon {-1.0}, mon2 {-1.0};

                        Habituation &h2 = morgan->getHabituation2();

                        bool habi2 =
                                h2.is_habituation ( vsum2, sum2, mon2 );

                        /*
                          std::cout  << "   HABI2 MONITOR:"
                              << m_internal_clock
                               << "[SEARCHING] MPU:" << mpu.first.c_str()
                           << vsum2 << sum2
                               << "bogocertainty of convergence:"
                               << mon2*100 << "%";
                        */

                        Habituation &h = morgan->getHabituation();
                        bool habi =
                                h.is_habituation ( vsum, sum, mon );

                        /*
                                  std::cout  << "   HABITUATION MONITOR:"
                                           << m_internal_clock
                                           << "[SEARCHING] MPU:" << mpu.first.c_str()
                                           << "bogocertainty of convergence:"
                                           << mon*100 << "%";
                        */

                        if ( habi2 || mon2 >= .9 ) { //.9 )
                                ++ell;

                                /*
                                              std::cout  << "   KNOWLEDGE MONITOR:"
                                                       << m_internal_clock
                                                       << "[DETECTED] MPU:" << mpu.first.c_str()
                                                       << "ELL" << ell;
                                                       */
                        }


                        if ( maxSamuQl == nullptr && ( habi || mon >= 1.0 ) ) { //.9 )
                                maxSamuQl = mpu.second;
                                //break;
                        }


                } // for MPUs

                /*
                if(ell)
                std::cout  << "   KNOWLEDGE MONITOR:";
                */

                // nem baj, ha sokáig kell menni, mert a párhuzamos szálakból a kiválasztott
                // folytatódik, a párhuzamosság a költség, meg ha nem talál, hanem új MPU kell...

                int t = m_internal_clock - m_searchingStart;

                if ( t > m_maxLearningTime || maxSamuQl ) {

                        if ( !maxSamuQl ) {
                                m_morgan = newMPU();

                                /*
                                              std::cout  << "   SENSITIZATION MONITOR:"
                                                       << m_internal_clock
                                                       << "MPU-notion:" << get_foobar ( ).c_str()
                                                       << "(new MPU, searching time)"
                                                       << t;
                                */
                        } else {
                                m_morgan = maxSamuQl;

                                /*
                                              std::cout  << "   SENSITIZATION MONITOR:"
                                                       << m_internal_clock
                                                       << "MPU-notion:" << get_foobar ( ).c_str()
                                                       << "(recognized MPU, searching time)"
                                                       << t;
                                */
                        }

                        init_MPUs ( true );

                        m_searching = false;
                        m_haveAlreadyLearnt = false;
                        m_haveAlreadyLearntTime = m_internal_clock;




                }

        }//searching
        else {

                //sum = pred ( reality, predictions, !searching, vsum ); //!haveAlreadyLearnt, vsum );
                sum = pred ( reality, center_of_tape, noc, predictions, m_haveAlreadyLearnt?5:0, vsum, sum2, vsum2 );

                double mon {-1.0};
                Habituation& h = m_morgan->getHabituation();
                m_habituation = h.is_habituation ( vsum, sum, mon );

                if ( !m_haveAlreadyLearnt ) {

                        /*
                                  std::cout  << "   HABITUATION MONITOR:"
                                           << m_internal_clock
                                           << "[LEARNING]"
                                           << get_foobar ( ).c_str()
                                           << "bogocertainty of convergence:"
                                           << mon*100 << "%";
                        */

                        if ( m_habituation ) {

                                m_haveAlreadyLearnt = true;

                                int t = m_internal_clock - m_haveAlreadyLearntTime;
                                if ( t > m_maxLearningTime ) {
                                        m_maxLearningTime = t;
                                }

                                /*              std::cout  << "   HIGHER-ORDER NOTION MONITOR:"
                                                       << m_internal_clock
                                                       << "MPU-notion:" << get_foobar ( ).c_str()
                                                       << "(learning time)"
                                                       << t;
                                */
                        }

                } else { // már "megtanulta"

                        /*
                                  std::cout  << "   HABITUATION MONITOR:"
                                           << m_internal_clock
                                           << "[LEARNED]"
                                           << get_foobar ( ).c_str()
                                           << "bogocertainty of convergence:"
                                           << mon*100 << "%";
                        */

                        if ( h.is_newinput ( vsum, sum ) ) { // && !m_habituation ) //&& mon != -1.0  /*&& mon != 1.0*/ )

                                if ( ++newc >=2 ) {
                                        /*
                                          std::cout  << "   SENSITIZATION MONITOR:"
                                                   << m_internal_clock
                                                   << "(new input detected)";
                                        */
                                        m_searching = true;
                                        m_searchingStart = m_internal_clock;

                                        init_MPUs ( false );
                                        newc = 0;
                                }
                        } else {
                                newc = 0;
                        }

                }

                * ( this->fp ) = m_morgan->getFp();
                * ( this->fr ) = m_morgan->getFr();


        }

}

void SamuBrain::init_MPUs ( bool ex )
{

        for ( auto& mpu : m_brain ) {

                MORGAN morgan = mpu.second;



                if ( ex ) {
                        if ( mpu.second != m_morgan ) {
                                morgan->getHabituation().clear();

                                morgan->getHabituation2().clear();

                        }
                } else {
                        morgan->getHabituation().clear();

                        morgan->getHabituation2().clear();

                }

                morgan->cls();

        }

}

std::string SamuBrain::get_foobar() const
{
        return get_foobar ( m_morgan );
}

std::string SamuBrain::get_foobar ( MORGAN samuQl ) const
{

        auto foobar = std::find_if (
                              std::begin ( m_brain ), std::end ( m_brain ),
        [=] ( auto&& mpu ) {
                return ( mpu.second ) == samuQl;
        }
                      );

        if ( foobar != std::end ( m_brain ) ) {
                return foobar->first;
        } else {
                return "Mystical knowledge";
        }
}


